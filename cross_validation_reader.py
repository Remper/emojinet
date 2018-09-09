import argparse

if __name__ == "__main__":
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description="Cross Validation results reader for EVALITA2018 ITAmoji task")

    parser.add_argument("--input-file",
                        default=None,
                        required=True,
                        help="Input file path")

    parser.add_argument("--output-file",
                        default=None,
                        required=True,
                        help="Output file path")

    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file

    evaluating = False
    with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
        for line in input_file:
            if line.startswith("INFO:root:Working on fold:"):
                output_file.write(line)
            if line.startswith("INFO:root:Evaluating"):
                evaluating = True
            if evaluating:
                if line.startswith("INFO:root:[      test]"):
                    evaluating = False
                    output_file.write(line)
