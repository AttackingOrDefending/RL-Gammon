def flipped_to_normal_moves(move_str, player=1):
    """
    Convert flipped-board move notation to normal-board PX-PY moves.

    Supports:
    1. Single bar move: "P0-1" -> "P25-P24 Pass"
    2. Chained moves: "P0-1-P19-1" -> "P25-P24 P19-P18"
    3. Multiple moves with same die: "2-P1-P12-P17-P19" -> "P24-P22 P13-P11 P8-P6 P6-P4"

    player: 1 or 2
        Player 1: Bar = P0, Bear off = P25
        Player 2: Bar = P25, Bear off = P0
    """

    def normal(p):
        return 25 - p

    # Check if first token is die value or P<number>
    tokens = move_str.strip().split("-")
    moves = []

    # Case 1: starts with a number (die for multiple moves)
    try:
        die = int(tokens[0])
        # Multiple moves: apply die to each remaining token
        for tok in tokens[1:]:
            tok = tok.upper().lstrip("P")
            start = int(tok)
            # Compute flipped end
            if start == 0 and player == 1:
                normal_start = 25
                normal_end = 25 - die
                # If bar move cannot be fully applied
                if normal_end < 1:
                    moves.append(f"P{normal_start}-P{normal_end} Pass")
                else:
                    moves.append(f"P{normal_start}-P{normal_end}")
            elif start == 25 and player == 2:
                normal_start = 0
                normal_end = die
                if normal_end > 24:
                    moves.append(f"P{normal_start}-P{normal_end} Pass")
                else:
                    moves.append(f"P{normal_start}-P{normal_end}")
            else:
                normal_start = normal(start)
                if player == 1:
                    normal_end = normal(start - die)
                else:
                    normal_end = normal(start + die)
                moves.append(f"P{normal_start}-P{normal_end}")
        return " ".join(moves)
    except ValueError:
        # Not a die value: treat as chained moves with distances
        i = 0
        while i < len(tokens) - 1:
            start_tok = tokens[i].upper().lstrip("P")
            end_tok = tokens[i + 1].upper().lstrip("P")
            start = int(start_tok)
            distance = int(end_tok)

            # Bar move
            if start == 0 and player == 1:
                normal_start = 25
                normal_end = 25 - distance
                if normal_end < 1:
                    moves.append(f"P{normal_start}-P{normal_end} Pass")
                else:
                    moves.append(f"P{normal_start}-P{normal_end}")
            elif start == 25 and player == 2:
                normal_start = 0
                normal_end = distance
                if normal_end > 24:
                    moves.append(f"P{normal_start}-P{normal_end} Pass")
                else:
                    moves.append(f"P{normal_start}-P{normal_end}")
            else:
                normal_start = normal(start)
                if player == 1:
                    normal_end = normal(start - distance)
                else:
                    normal_end = normal(start + distance)
                moves.append(f"P{normal_start}-P{normal_end}")
            i += 2  # Move to next pair
        return " ".join(moves)

move = "P4-1-P19-1"
player = 1

print(flipped_to_normal_moves(move, player))

