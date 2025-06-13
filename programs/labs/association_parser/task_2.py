import pandas as pd
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
import ollama
import time


def get_relations(stimulus, reaction):
    try:
        # Get syntagmatic relation
        syntagmatic_prompt = f"Какое синтагматическое отношение между словами '{stimulus}' и '{reaction}'? Ответь одним словом или короткой фразой."
        syntagmatic_response = ollama.generate(
            model="mistral:7b",
            prompt=syntagmatic_prompt,
            options={"temperature": 0.7},
        )
        syntagmatic = syntagmatic_response["response"].strip().split("\n")[-1]

        # Get paradigmatic relation
        paradigmatic_prompt = f"Какое парадигматическое отношение между словами '{stimulus}' и '{reaction}'? Ответь одним словом или короткой фразой."
        paradigmatic_response = ollama.generate(
            model="mistral:7b",
            prompt=paradigmatic_prompt,
            options={"temperature": 0.7},
        )
        paradigmatic = paradigmatic_response["response"].strip().split("\n")[-1]

        print(syntagmatic, paradigmatic)

        return syntagmatic, paradigmatic
    except Exception as e:
        print(f"Error getting relations for {stimulus}-{reaction}: {str(e)}")
        return "Error", "Error"


def process_excel_files():
    # Create a new Excel writer
    output_file = "processed_results.xlsx"
    writer = pd.ExcelWriter(output_file, engine="openpyxl")

    # Process each Excel file in the documents directory
    for filename in os.listdir("documents"):
        if filename.endswith("306.XLS") or filename.endswith("406.XLS"):
            file_path = os.path.join("documents", filename)
            print(f"Processing {filename}...")

            try:
                # Read the Excel file with xlrd engine for old .xls files
                df = pd.read_excel(file_path, engine="xlrd")

                # Assuming the columns are named 'Stim' and 'Reak'
                if "Stim" in df.columns and "Reak" in df.columns:
                    # Create new columns for relations
                    df["Синтагматическое отношение"] = ""
                    df["Парадигматическое отношение"] = ""

                    # Process each row
                    for idx, row in df.iterrows():
                        stimulus = str(row["Stim"])
                        reaction = str(row["Reak"])
                        # Skip empty or invalid entries
                        if (
                            pd.isna(stimulus)
                            or pd.isna(reaction)
                            or stimulus == "nan"
                            or reaction == "nan"
                        ):
                            continue

                        print(
                            f"Processing pair {idx+1}/{len(df)}: {stimulus}-{reaction}"
                        )

                        # Get relations from the model
                        syntagmatic, paradigmatic = get_relations(stimulus, reaction)

                        # Update the dataframe
                        df.at[idx, "Синтагматическое отношение"] = syntagmatic
                        df.at[idx, "Парадигматическое отношение"] = paradigmatic

                        # Add a small delay to avoid overwhelming the model
                        time.sleep(0.5)

                    # Rename columns and select only needed ones
                    df = df.rename(columns={"Stim": "Стимул", "Reak": "Реакция"})

                    # Select only needed columns in the correct order
                    df = df[
                        [
                            "Стимул",
                            "Реакция",
                            "Синтагматическое отношение",
                            "Парадигматическое отношение",
                        ]
                    ]

                    # Save to the output Excel file as a new sheet
                    sheet_name = os.path.splitext(filename)[0]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    # Set column width to 330 pixels (approximately 45 characters)
                    worksheet = writer.sheets[sheet_name]
                    for idx, col in enumerate(df.columns):
                        column_letter = get_column_letter(idx + 1)
                        worksheet.column_dimensions[column_letter].width = (
                            45  # 330 pixels ≈ 45 characters
                        )

                    print(f"Completed processing {filename}")
                else:
                    print(f"Warning: Required columns not found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Save the final Excel file
    writer.close()
    print(f"Processing complete. Results saved to {output_file}")


if __name__ == "__main__":
    process_excel_files()
