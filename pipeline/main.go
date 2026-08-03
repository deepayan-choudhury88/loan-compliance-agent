package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var countryToCurrency = map[string]string{
	"Germany": "EUR", "Ireland": "EUR", "Spain": "EUR", "France": "EUR",
	"Italy": "EUR", "Netherlands": "EUR", "Portugal": "EUR", "Belgium": "EUR",
	"Austria": "EUR", "Finland": "EUR", "Luxembourg": "EUR",
	"United Kingdom": "GBP", "United States": "USD", "Switzerland": "CHF",
	"Sweden": "SEK", "Poland": "PLN", "Japan": "JPY", "Canada": "CAD",
}

type ExchangeRates struct {
	Rates map[string]float64 `json:"rates"`
}

type LoanInput struct {
	LoanID           string  `json:"loan_id"`
	CompanyName      string  `json:"company_name"`
	HQCountry        string  `json:"hq_country"`
	AssetDescription string  `json:"asset_description"`
	AssetValue       float64 `json:"asset_value"`
	AssetOwner       string  `json:"asset_owner"`
	LoanValue        float64 `json:"loan_value"`
	LoanCurrency     string  `json:"loan_currency"`
	LoanValueEUR     float64 `json:"loan_value_eur"`
	ExpectedCurrency string  `json:"expected_currency"`
}

type OPAResponse struct {
	Result struct {
		Allow      bool     `json:"allow"`
		Violations []string `json:"violations"`
	} `json:"result"`
}

type ReportData struct {
	TotalLoansChecked int
	TotalFailures     int
	Rule1Fails        int
	Rule2Fails        int
	Rule3Fails        int
	Portfolio         map[string]float64
	FailedLoans       []FailedLoan
}

type FailedLoan struct {
	LoanID      string
	Company     string
	Violations  string
	LoanDetails string
}

func fetchRates() (map[string]float64, error) {
	client := http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get("https://api.frankfurter.dev/v1/latest")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var data ExchangeRates
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	return data.Rates, nil
}

func main() {
	rates, err := fetchRates()
	if err != nil {
		log.Fatalf("Failed to fetch exchange rates: %v", err)
	}
	rates["EUR"] = 1.0

	file, err := os.Open("../data/loans.csv")
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err = reader.Read(); err != nil {
		log.Fatalf("Failed to read header: %v", err)
	}

	report := ReportData{
		Portfolio:   make(map[string]float64),
		FailedLoans: []FailedLoan{},
	}

	// CONCURRENCY UPGRADES:
	var wg sync.WaitGroup
	var mu sync.Mutex
	// Semaphore to limit concurrent HTTP requests to 50 so we don't crash OPA
	sem := make(chan struct{}, 50) 

	// Reusable HTTP client so we don't exhaust local ports
	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 50,
		},
		Timeout: 5 * time.Second,
	}

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue
		}

		// Parse row variables safely outside the goroutine
		loanID := record[0]
		companyName := record[1]
		hqCountry := record[2]
		assetDesc := record[3]
		assetValue, _ := strconv.ParseFloat(record[4], 64)
		assetOwner := record[5]
		loanValue, _ := strconv.ParseFloat(record[6], 64)
		currency := record[7]

		var loanValueEUR float64
		if currency == "EUR" {
			loanValueEUR = loanValue
		} else {
			rate, exists := rates[currency]
			if !exists { continue }
			loanValueEUR = loanValue / rate
		}

		expectedCurrency, exists := countryToCurrency[hqCountry]
		if !exists {
			expectedCurrency = "UNKNOWN"
		}

		// Start a Goroutine for this row
		wg.Add(1)
		sem <- struct{}{} // Claim a slot in the semaphore (max 50)

		go func(lID, cName, hq, aDesc, aOwner, cur, expCur string, aVal, lVal, lValEUR float64) {
			defer wg.Done()
			defer func() { <-sem }() // Release the slot when done

			loanInput := LoanInput{
				LoanID:           lID,
				CompanyName:      cName,
				HQCountry:        hq,
				AssetDescription: aDesc,
				AssetValue:       aVal,
				AssetOwner:       aOwner,
				LoanValue:        lVal,
				LoanCurrency:     cur,
				LoanValueEUR:     lValEUR,
				ExpectedCurrency: expCur,
			}

			reqBody := map[string]interface{}{"input": loanInput}
			jsonData, _ := json.Marshal(reqBody)

			resp, err := client.Post("http://localhost:8181/v1/data/compliance", "application/json", bytes.NewBuffer(jsonData))
			if err != nil { return }

			var opaResult OPAResponse
			if err := json.NewDecoder(resp.Body).Decode(&opaResult); err != nil {
				resp.Body.Close()
				return
			}
			resp.Body.Close()

			// MUTEX LOCK: Safely update the shared report variables
			mu.Lock()
			report.Portfolio[cName] += lValEUR
			report.TotalLoansChecked++

			if !opaResult.Result.Allow {
				report.TotalFailures++
				violationsStr := strings.Join(opaResult.Result.Violations, "; ")
				
				if strings.Contains(violationsStr, "Rule 1") { report.Rule1Fails++ }
				if strings.Contains(violationsStr, "Rule 2") { report.Rule2Fails++ }
				if strings.Contains(violationsStr, "Rule 3") { report.Rule3Fails++ }

				details := fmt.Sprintf("Loan Value: %.2f %s | Asset Value: %.2f %s | HQ: %s", lVal, cur, aVal, cur, hq)

				report.FailedLoans = append(report.FailedLoans, FailedLoan{
					LoanID:      lID,
					Company:     cName,
					Violations:  violationsStr,
					LoanDetails: details,
				})
			}
			mu.Unlock() // MUTEX UNLOCK
		}(loanID, companyName, hqCountry, assetDesc, assetOwner, currency, expectedCurrency, assetValue, loanValue, loanValueEUR)
	}

	fmt.Println("Processing all rows concurrently... please wait.")
	wg.Wait() // Wait for all 100,000 goroutines to finish

	generateHTMLReport(report)
}

func generateHTMLReport(data ReportData) {
	htmlTemplate := `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Compliance Review Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 text-gray-800 font-sans p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-bold mb-8 text-gray-900">Loan Compliance Report</h1>
        
        <div class="grid grid-cols-4 gap-4 mb-8">
            <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <h3 class="text-sm font-medium text-gray-500 uppercase">Total Checked</h3>
                <p class="text-3xl font-bold">{{.TotalLoansChecked}}</p>
            </div>
            <div class="bg-red-50 p-6 rounded-lg shadow-sm border border-red-100">
                <h3 class="text-sm font-medium text-red-500 uppercase">Rule 1 Failures</h3>
                <p class="text-3xl font-bold text-red-700">{{.Rule1Fails}}</p>
            </div>
            <div class="bg-orange-50 p-6 rounded-lg shadow-sm border border-orange-100">
                <h3 class="text-sm font-medium text-orange-500 uppercase">Rule 2 Failures</h3>
                <p class="text-3xl font-bold text-orange-700">{{.Rule2Fails}}</p>
            </div>
            <div class="bg-yellow-50 p-6 rounded-lg shadow-sm border border-yellow-100">
                <h3 class="text-sm font-medium text-yellow-500 uppercase">Rule 3 Failures</h3>
                <p class="text-3xl font-bold text-yellow-700">{{.Rule3Fails}}</p>
            </div>
        </div>

        <div class="bg-white rounded-lg shadow-sm border border-gray-200 mb-8 overflow-hidden">
            <div class="px-6 py-4 border-b border-gray-200 bg-gray-50">
                <h2 class="text-lg font-semibold text-gray-800">Action Required: Compliance Failures</h2>
            </div>
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Loan ID</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Company</th>
                        <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Violations</th>
                        <th class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {{range .FailedLoans}}
                    <tr id="row-{{.LoanID}}" class="hover:bg-gray-50 transition-colors">
                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{{.LoanID}}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{.Company}}</td>
                        <td class="px-6 py-4 text-sm text-red-600">{{.Violations}}</td>
                        <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <button id="btn-{{.LoanID}}" class="text-indigo-600 font-bold hover:text-indigo-900 mr-4" onclick="getAIFix('{{.LoanID}}', '{{.Company}}', '{{.Violations}}', '{{.LoanDetails}}')">Get AI Fix</button>
                            <button class="text-gray-400 hover:text-gray-600 font-bold mr-4" onclick="resolveLoan('{{.LoanID}}', 'ignored')">Ignore</button> 
                            <button class="text-green-600 hover:text-green-900 font-bold" onclick="resolveLoan('{{.LoanID}}', 'fixed')">Fix</button>
                        </td>
                    </tr>
                    {{end}}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function getAIFix(loanID, company, violations, loanDetails) {
            const btn = document.getElementById('btn-' + loanID);
            const tr = document.getElementById('row-' + loanID);
            
            const originalText = btn.innerText;
            btn.innerText = 'Analyzing...';
            btn.disabled = true;

            // Remove existing result row if it was clicked before
            const existingRes = document.getElementById('res-' + loanID);
            if (existingRes) existingRes.remove();

            try {
                const response = await fetch('http://localhost:5000/api/remediate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        company_name: company,
                        violations: violations,
                        loan_details: loanDetails
                    })
                });
                const data = await response.json();
                
                // Create a new row to display the AI response gracefully
                const newRow = document.createElement('tr');
                newRow.id = 'res-' + loanID;
                newRow.className = 'bg-indigo-50 border-l-4 border-indigo-500';
                
                // Convert simple Markdown bolding and newlines to HTML
                const formattedText = data.remediation
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br/>');
                
                newRow.innerHTML = '<td colspan="4" class="px-6 py-4 text-sm text-gray-800">' + formattedText + '</td>';
                tr.parentNode.insertBefore(newRow, tr.nextSibling);

            } catch (error) {
                alert("Error connecting to Python backend: " + error);
            } finally {
                btn.innerText = 'AI Fix Ready';
                btn.disabled = false;
            }
        }

        function resolveLoan(loanID, action) {
            const row = document.getElementById('row-' + loanID);
            const aiRow = document.getElementById('res-' + loanID);
            
            if (action === 'fixed') {
                row.className = 'bg-green-50 transition-colors opacity-75';
                row.querySelector('td:nth-child(3)').innerHTML = '<span class="text-green-700 font-bold">✓ Manually Fixed</span>';
            } else if (action === 'ignored') {
                row.className = 'bg-gray-100 transition-colors opacity-50';
                row.querySelector('td:nth-child(3)').innerHTML = '<span class="text-gray-500 font-bold">∅ Ignored</span>';
            }
            
            // Remove the action buttons so it can't be clicked again
            row.querySelector('td:nth-child(4)').innerHTML = '<span class="text-gray-500 text-sm">Resolved</span>';
            
            // Hide the AI suggestion row if it was open
            if (aiRow) {
                aiRow.style.display = 'none';
            }
        }
    </script>
</body>
</html>`

	tmpl, err := template.New("report").Parse(htmlTemplate)
	if err != nil {
		log.Fatalf("Template parsing failed: %v", err)
	}

	file, err := os.Create("compliance_report.html")
	if err != nil {
		log.Fatalf("Failed to create HTML report: %v", err)
	}
	defer file.Close()

	if err := tmpl.Execute(file, data); err != nil {
		log.Fatalf("Failed to execute template: %v", err)
	}

	fmt.Println("✅ Success! Interactive report generated at: compliance_report.html")
}
