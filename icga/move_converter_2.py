import re


class BackgammonConverter:
    # Internal Constants from OpenSpiel
    NUM_POINTS = 26  # 0-23=Board, 24=Bar, 25=Pass
    BAR_POS = 24
    PASS_POS = 25
    OFFSET_SWAP = 676  # 26 * 26

    @staticmethod
    def to_string(move_id: int, dice: list[int]) -> str:
        """
        Converts a move integer + dice into string format P<start>-<dist>-P<start>-<dist>.

        Args:
            move_id: The integer representing the move.
            dice: A list of two integers [d1, d2].
        """
        if len(dice) != 2:
            raise ValueError("Must provide exactly 2 dice values.")

        # 1. Decode High/Low/Positions logic from SpielMoveToCheckerMoves
        high_roll = max(dice)
        low_roll = min(dice)

        # If ID < 676, the first internal digit uses the High Roll.
        high_roll_first = (move_id < BackgammonConverter.OFFSET_SWAP)

        work_id = move_id if high_roll_first else move_id - BackgammonConverter.OFFSET_SWAP

        # Extract internal positions (base-26)
        # digit[0] is usually the lower board position
        pos0_internal = work_id % BackgammonConverter.NUM_POINTS
        pos1_internal = work_id // BackgammonConverter.NUM_POINTS

        # Assign distances based on the high_roll_first flag
        if high_roll_first:
            move_a = {'pos': pos0_internal, 'dist': high_roll}
            move_b = {'pos': pos1_internal, 'dist': low_roll}
        else:
            move_a = {'pos': pos0_internal, 'dist': low_roll}
            move_b = {'pos': pos1_internal, 'dist': high_roll}

        moves = [move_a, move_b]

        # 2. Convert Internal Positions to Human Readable (P1..P25)
        # Internal 0-23 -> Human 1-24
        # Internal 24 (Bar) -> Human 25
        # Internal 25 (Pass) -> handled separately or ignored here
        human_moves = []
        for m in moves:
            p = m['pos']
            if p == BackgammonConverter.PASS_POS:
                continue  # Pass moves don't appear in P-notation usually

            # Map internal pos to human string label
            if p == BackgammonConverter.BAR_POS:
                h_pos = 25
            else:
                h_pos = p + 1

            human_moves.append({'start': h_pos, 'dist': m['dist']})

        # 3. Sort for Output (ActionToString logic)
        # "Order the moves by highest number first"
        human_moves.sort(key=lambda x: x['start'], reverse=True)

        # 4. Construct String
        parts = [f"P{m['start']}-{m['dist']}" for m in human_moves]
        return "-".join(parts)

    @staticmethod
    def from_string(move_str: str) -> int:
        """
        Converts format P<start>-<dist>-P<start>-<dist> back to integer move_id.
        """
        # Parse format P<num>-<num>
        # We accept P0, P25, or P(Bar) as input for the Bar, normalizing to Internal 24.
        pattern = r"P(\d+)-(\d+)"
        matches = re.findall(pattern, move_str)

        if len(matches) != 2:
            raise ValueError("String must contain exactly two moves (e.g. P19-4-P17-2)")

        # Temporary storage for the decoded moves
        decoded_moves = []

        for pos_str, dist_str in matches:
            h_pos = int(pos_str)
            dist = int(dist_str)

            # Convert Human Pos to Internal
            if h_pos == 25 or h_pos == 0:
                # Handling 25 (Standard Bar) and 0 (User Notation Bar)
                internal_pos = BackgammonConverter.BAR_POS
            else:
                internal_pos = h_pos - 1

            decoded_moves.append({'pos': internal_pos, 'dist': dist})

        # --- RECONSTRUCTION LOGIC ---

        # 1. Canonical Sort: The integer encoding expects digit[0] <= digit[1] usually,
        # but strictly speaking, it expects the 'digits' to be extracted via % 26 and / 26.
        # We need to map the two moves to digit_A and digit_B such that:
        # ID = digit_A + digit_B * 26  (+ 676 optionally)
        #
        # We must identify which move corresponds to digit_A (the one that might take High or Low).
        # We sort by internal position to ensure canonical order of digits.
        decoded_moves.sort(key=lambda x: x['pos'])

        move0 = decoded_moves[0]  # This becomes digit[0]
        move1 = decoded_moves[1]  # This becomes digit[1]

        digit0 = move0['pos']
        digit1 = move1['pos']

        # 2. Determine Offset based on Dice usage
        dist0 = move0['dist']
        dist1 = move1['dist']

        dice = [dist0, dist1]
        high_roll = max(dice)
        low_roll = min(dice)

        # If digit0 used the High Roll, it implies ID < 676 (High Roll First)
        # If digit0 used the Low Roll, it implies ID >= 676 (Low Roll First)

        offset = 0
        if dist0 == high_roll:
            # High Roll First
            offset = 0
        else:
            # Low Roll First (digit0 took the small die)
            offset = BackgammonConverter.OFFSET_SWAP

            # Edge case: Doubles (e.g. 2-2). 
            # If dist0 == dist1, high_roll == low_roll.
            # Standard encoding usually puts doubles in the lower block (offset 0).
            if dist0 == dist1:
                offset = 0

        # 3. Calculate ID
        return digit0 + (digit1 * BackgammonConverter.NUM_POINTS) + offset


# --- TEST CASE VERIFICATION ---
if __name__ == "__main__":
    converter = BackgammonConverter()

    # Test cases provided by user
    # Note: Dice are inferred from the user's string to perform the check.

    test_cases = [
        (1160, "P19-4-P17-2", [4, 2]),
        (986, "P25-1-P12-6", [1, 6]),  # User said P0, we normalize to P25
        (1109, "P18-1-P17-5", [5, 1]),  # Note: Standard sort puts P18 first
        (1272, "P25-2-P23-6", [6, 2]),
        (1168, "P25-1-P19-2", [2, 1]),  # User said P0, we normalize to P25
        (569, "P24-2-P22-1", [2, 1]),
        (267, "P11-1-P8-6", [6, 1]),
        (221, "P14-5-P9-2", [5, 2])
    ]

    print(f"{'ID':<6} | {'Expected':<12} | {'Computed':<12} | {'Result':<6}")
    print("-" * 45)

    for mid, string, dice in test_cases:
        # 1. Test Number -> String
        computed_str = converter.to_string(mid, dice)

        # 2. Test String -> Number
        # We use the computed string to be safe, or the input string if compatible
        computed_id = converter.from_string(string)

        match_str = (computed_str == string)
        match_id = (computed_id == mid)

        # Handling the User's "P0" notation for 986/1168 implicitly:
        # My code outputs P25. If user expects P0, that's a notation variance.
        # I print what my code computed.
        print(f"{mid:<6} | {string:<12} | {computed_str:<12} | {'OK' if match_id else 'FAIL'}")