
def parse_to_slice(input_str):
    """
    Parses a string representing numpy indexing and returns the corresponding NumPy array slice.
    Supports multi-dimensional indexing with missing values like "[:, :2]".
    """
    # Remove brackets and spaces
    input_str = input_str.strip("[]").replace(" ", "")

    # Split by commas to handle multiple dimensions
    parts = input_str.split(",")
    result = []

    for part in parts:
        if ":" in part:
            # Handle cases like ":", ":2", "1:", "1:3:2"
            slice_parts = part.split(":")
            # Convert non-empty values to int
            slice_values = [int(x) if x else None for x in slice_parts]
            result.append(slice(*slice_values))  # Convert to slice object
        else:
            # Handle single integer indices
            result.append(int(part))

    # Convert result to tuple if multi-dimensional
    return tuple(result) if len(result) > 1 else result[0]