import pyspiel

import re

def slash_to_px_distance(move_str, player=1):
    move_str = move_str.replace("Pass", "")
    move_str = move_str.replace("Bar", "25")
    move_str = move_str.replace("Off", "0")
    # Keep only allowed tokens
    move_str = re.sub(r"[^0-9/BarOff ]", "", move_str, flags=re.IGNORECASE)

    if move_str == "":
        return "pass"

    result = []

    # Split separate move groups
    for part in move_str.strip().split():
        tokens = part.split("/")

        # Handle Bar
        if tokens[0].lower() == "bar":
            start = 25 if player == 1 else 0
            end = int(tokens[1])
            distance = abs(start - end)
            result.append(f"P{start}-{distance}")
            continue

        # Numeric chains
        points = [int(t) for t in tokens]
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            distance = abs(start - end)
            result.append(f"P{start}-{distance}")

    return "-".join(result)



if __name__ == "__main__":
    game = pyspiel.load_game("backgammon(scoring_type=full_scoring)")
    state = game.new_initial_state()
    state.apply_action(1)

    actions = state.legal_actions(state.current_player())
    for action in actions:
        action_string = state.action_to_string(state.current_player(), action)
        s = action_string.split(" - ")[1]
        print(s)
        print(slash_to_px_distance(s))
        print("")
