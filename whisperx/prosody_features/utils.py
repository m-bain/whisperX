def generate_char_frame_sequence(char_data, null_token: str ='<NULL>'):
    r"""
    Generate a list of characters for each frame

    Args:
        char_data (list of dict): List of dictionaries with 'char', 'start', and 'end'. Produced by alignmnet function's 'char' output.
        null_token (str): token to assing for non-speech frames. Defaults to '<NULL>'.
        
    Returns:
        list of str: List of characters for each frame.
    """
    
    # Standardize to lowercase
    char_data = [{**entry, 'char': entry['char'].lower()} for entry in char_data if 'start' in entry]
    
    # Determine the total time range from the min start and max end
    min_time = int(min(entry['start'] for entry in char_data))
    max_time = int(max(entry['end'] for entry in char_data))
    
    # Initialize the output list with the <NULL> token
    time_char_sequence = [null_token for _ in range(min_time, max_time)]
    
    # Assign characters to time intervals
    for entry in char_data:
        char = entry['char']
        start = int(entry['start']) - min_time
        end = int(entry['end']) - min_time
        for t in range(start, end):
            time_char_sequence[t] = char if char.strip() else null_token

    return time_char_sequence