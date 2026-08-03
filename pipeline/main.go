package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
)

// LoanRecord maps directly to the columns in loans.csv
type LoanRecord struct {
	LoanID           string
	CompanyName      string
	HQCountry        string
	AssetDescription string
	AssetValue       string // We will parse these to floats later for math
	AssetOwner       string
	LoanValue        string
	LoanCurrency     string
}

func main() {
	// 1. Open the file
	file, err := os.Open("../data/loans.csv")
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	// 2. Initialize the CSV reader
	reader := csv.NewReader(file)

	// Read and discard the header row
	_, err = reader.Read()
	if err != nil {
		log.Fatalf("Failed to read header: %v", err)
	}

	// 3. Iterate through the records
	rowCount := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break // End of file
		}
		if err != nil {
			log.Printf("Error reading row: %v", err)
			continue
		}

		// Map the raw string slice to our struct
		loan := LoanRecord{
			LoanID:           record[0],
			CompanyName:      record[1],
			HQCountry:        record[2],
			AssetDescription: record[3],
			AssetValue:       record[4],
			AssetOwner:       record[5], // Reminder: We ignore this column for rules
			LoanValue:        record[6],
			LoanCurrency:     record[7],
		}

		// Print the first row just to verify it works
		if rowCount == 0 {
			fmt.Printf("Successfully read first record: %+v\n", loan)
		}
		rowCount++
	}

	fmt.Printf("Successfully parsed %d rows.\n", rowCount)
}
