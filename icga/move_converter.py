import re


class BackgammonConverter:
    # Internal Constants
    NUM_POINTS = 26  # 0-23 (board), 24 (Bar), 25 (Pass)
    OFFSET_SWAP = 676  # 26 * 26
    BAR_POS_INTERNAL = 24

    @staticmethod
    def decode(move_id: int, dice: list[int]) -> str:
        """
        Converts integer ID to format P<pos>-<dist>-P<pos>-<dist>.
        """
        if len(dice) != 2:
            raise ValueError("Dice list must contain exactly 2 integers.")

        # 1. Determine High/Low rolls and Order
        high_roll = max(dice)
        low_roll = min(dice)

        # If move_id < 676, First Logical Move uses High Roll.
        # If move_id >= 676, First Logical Move uses Low Roll.
        high_roll_first = (move_id < BackgammonConverter.OFFSET_SWAP)

        # Normalize ID to extract positions
        work_id = move_id if high_roll_first else move_id - BackgammonConverter.OFFSET_SWAP

        # 2. Extract Internal Positions (digits[0] is Move 0, digits[1] is Move 1)
        # digit[0] is the lower 26-base digit, digit[1] is the upper.
        pos_0_internal = work_id % BackgammonConverter.NUM_POINTS
        pos_1_internal = work_id // BackgammonConverter.NUM_POINTS

        # 3. Assign Distances based on Logical Order
        if high_roll_first:
            move0 = {'pos': pos_0_internal, 'dist': high_roll}
            move1 = {'pos': pos_1_internal, 'dist': low_roll}
        else:
            move0 = {'pos': pos_0_internal, 'dist': low_roll}
            move1 = {'pos': pos_1_internal, 'dist': high_roll}

        moves = [move0, move1]

        # 4. Helper to Convert Internal Pos to Human String (P1..P25)
        def to_human_pos(p):
            if p == BackgammonConverter.BAR_POS_INTERNAL:
                return 25  # Bar usually represented as 25 in simple P-notation
            return p + 1

        # 5. SORT for Display: Highest Start Position First
        # The C++ ActionToString forces the move with the higher start pos
        # to be displayed first in the string.
        moves.sort(key=lambda x: x['pos'], reverse=True)

        return f"P{to_human_pos(moves[0]['pos'])}-{moves[0]['dist']}-P{to_human_pos(moves[1]['pos'])}-{moves[1]['dist']}"

    @staticmethod
    def encode(move_str: str) -> int:
        """
        Converts format P<pos>-<dist>-P<pos>-<dist> back to Integer ID.
        Reconstructs canonical ID assuming digit[0] <= digit[1].
        """
        # 1. Parse String
        pattern = r"P(\d+)-(\d+)-P(\d+)-(\d+)"
        match = re.search(pattern, move_str)
        if not match:
            raise ValueError(f"Invalid format: {move_str}")

        h_pos1, dist1, h_pos2, dist2 = map(int, match.groups())

        # Helper to Convert Human Pos to Internal
        def to_internal(p):
            if p == 25: return BackgammonConverter.BAR_POS_INTERNAL
            return p - 1

        # Create Move Objects
        moves = [
            {'pos': to_internal(h_pos1), 'dist': dist1},
            {'pos': to_internal(h_pos2), 'dist': dist2}
        ]

        # 2. CANONICALIZE: Sort by Position (Ascending)
        # The integer encoding (pos0 + pos1*26) is typically constructed
        # such that digit[0] is the smaller position to ensure uniqueness.
        moves.sort(key=lambda x: x['pos'])

        # Now moves[0] corresponds to digits[0] (Pos 0)
        # Now moves[1] corresponds to digits[1] (Pos 1)
        pos0 = moves[0]['pos']
        dist0 = moves[0]['dist']

        pos1 = moves[1]['pos']
        dist1 = moves[1]['dist']

        # 3. Determine Dice and Offset
        # We need to check if the Logical First Move (moves[0]) used the High or Low die.
        dice = [dist0, dist1]
        high_roll = max(dice)
        low_roll = min(dice)

        if dist0 == high_roll:
            # Move 0 used the High Roll -> High Roll First -> Offset 0
            offset = 0
        else:
            # Move 0 used the Low Roll -> Low Roll First -> Offset 676
            offset = BackgammonConverter.OFFSET_SWAP

        # 4. Calculate ID
        # ID = digits[0] + (digits[1] * 26) + offset
        encoded_id = pos0 + (pos1 * BackgammonConverter.NUM_POINTS) + offset

        return encoded_id


# --- Verification ---
if __name__ == "__main__":
    converter = BackgammonConverter()

    print("--- Test Case 1: The User's Example ---")
    # Expected: 1160
    # Logic:
    # String P19-4... means Human Pos 19 and 17. Internal 18 and 16.
    # Canonical sort puts 16 first.
    # Move 0 is Pos 16. In the string, P17 (16) moved 2.
    # Move 0 Dist is 2. Dice are 4, 2.
    # Move 0 used Low Roll (2). Offset = 676.
    # ID = 16 + (18 * 26) + 676 = 16 + 468 + 676 = 1160.
    s1 = "P0-1-P12-6"
    id1 = converter.encode(s1)
    print(f"String:  {s1}")
    print(f"Encoded: {id1} (Expected 1160)")

    # Reverse
    dice1 = [4, 2]
    out1 = converter.decode(id1, dice1)
    print(f"Decoded: {out1} (Expected P19-4-P17-2)")
    print("-" * 30)

    print("--- Test Case 2: High Roll First Example ---")
    # Let's try internal 16(dist 4) and 18(dist 2).
    # Move 0 (16) uses High(4). Offset 0.
    # ID = 16 + (18*26) = 484.
    # String should still sort high pos first: P19-2-P17-4.
    id2 = 484
    dice2 = [4, 2]
    out2 = converter.decode(id2, dice2)
    print(f"ID:      {id2}")
    print(f"Decoded: {out2}")

    rec2 = converter.encode(out2)
    print(f"Re-Enc:  {rec2} (Expected 484)")

