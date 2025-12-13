from rlgammon.agents.td_agent import TDAgent


class ICGAAgent(TDAgent):
    """Agent class for use in the ICGA competition."""

    def __init__(self, pre_made_model_file_name: str | None = None) -> None:
        """Construct the agent by loading a pre-trained model."""
        super().__init__(pre_made_model_file_name)


def action_to_discord(action: int, dice, action2=None) -> str:
    """
    Return moves of the format:
    Chance events (dice throws) are communicated by the referee using the format “<dice_value1>-<dice_value2>”, for example “5-6”.
    If no movement is possible, the player’s action must be “pass”.
    The standard action consists of moving two pieces. Its format is "<position1>-
    <dice1>-<position2>-<dice2>". For example, if the dice values at the start of
    the game are 5 and 6, one legal action is P1-6-P12-5 (see the backgammon figure). This action moves a piece from position P1 by 6 points (using die 6),
    then a piece from position P12 by 5 points (using die 5).
    If only a single piece can be moved, the format is simply "<position1>-
    <dice1>". For example : “P19-6”.
    In the case of doubles, the action format is one of the following : “<dice>-
    <position1>-<position2>-<position3>-<position4>” or “<dice>-<position1>-
    <position2>-<position3>”” or “<dice>-<position1>-<position2>” or “<dice>-
    <position1>”. For example, “2-P1-P12-P17-P19”.
    The bar position for the first player is P0. The bar position for the second
    player is P25.
    27
    Additional explanations : There is "P" if the next number is a position. P2
    is the position ("point") 2. See the first backgammon figure of game_move_coordinate.pdf
    for all available positions : "P0" to "P25".
    If there is no "P", it is a dice value.
    I assume a piece at the bar. Move “25-22” is incorrect but “P25-3” is correct.
    Move “0-3” is incorrect but “P0-3” is correct.
    I assume a piece to bear off. Move “4-0” is incorrect but “P4-4” is correct.
    Move “21-25” is incorrect, but “P21-4” is correct.
    I assume a piece to normal move. Move “13-17” is incorrect but “P13-4” is
    correct.
    During your turn, if you make "P4-4" then "P13-5", write "P4-4-P13-5".
    If you have the same dice twice, for instance "2-2", your move is for instance
    "2-P0-P2-P4-P6".
    Example of a game start :
    1. chance : 3-1
    2. player 0 : P19-3-P19-1
    3. chance : 3-2
    4. player 1 : P24-2-P6-3
    5. chance : 2-2
    6. player 0 : 2-P0-P2-P19-P19
    """
    if action == 0:
        return 'pass'
    """
    std::vector<CheckerMove> BackgammonState::SpielMoveToCheckerMoves(int player, Action spiel_move) const {
      SPIEL_CHECK_GE(spiel_move, 0);
      SPIEL_CHECK_LT(spiel_move, kNumDistinctActions);
    
      bool high_roll_first = spiel_move < 676;
      if (!high_roll_first) {
        spiel_move -= 676;
      }
    
      std::vector<Action> digits = {spiel_move % 26, spiel_move / 26};
      std::vector<CheckerMove> cmoves;
      int high_roll = DiceValue(0) >= DiceValue(1) ? DiceValue(0) : DiceValue(1);
      int low_roll = DiceValue(0) < DiceValue(1) ? DiceValue(0) : DiceValue(1);
    
      for (int i = 0; i < 2; ++i) {
        SPIEL_CHECK_GE(digits[i], 0);
        SPIEL_CHECK_LE(digits[i], 25);
    
        int num = -1;
        if (i == 0) {
          num = high_roll_first ? high_roll : low_roll;
        } else {
          num = high_roll_first ? low_roll : high_roll;
        }
        SPIEL_CHECK_GE(num, 1);
        SPIEL_CHECK_LE(num, 6);
    
        if (digits[i] == EncodedPassMove()) {
          cmoves.push_back(CheckerMove(kPassPos, -1, false));
        } else {
          cmoves.push_back(CheckerMove(
              digits[i] == EncodedBarMove() ? kBarPos : digits[i], num, false));
        }
      }
    
      return cmoves;
    }
    """
    high_roll_first = action < 676
    if not high_roll_first:
        action -= 676
    digits = [action % 26, action // 26]
    moves = []
    high_roll = max(dice)
    low_roll = min(dice)
    for i in range(2):
        if i == 0:
            num = high_roll if high_roll_first else low_roll
        else:
            num = low_roll if high_roll_first else high_roll
        if digits[i] == 25:
            moves.append(f'P{25}-{num}')
        else:
            moves.append(f'P{digits[i]}-{num}')
    if len(dice) > 2 and action2 is not None:
        # doubles
        moves_extra = []
        for i in range(2):
            if i == 0:
                num = dice[0]
            else:
                num = dice[0]
            if action2[i] == 25:
                moves_extra.append(f'P{25}-{num}')
            else:
                moves_extra.append(f'P{action2[i]}-{num}')
        return f"{dice[0]}-" + "-".join(moves + moves_extra)
    return "-".join(moves)


agent = ICGAAgent("../rlgammon/agents/saved_agents/td-backgammon-7f1110e4-5d23-461f-8294-731ee36a71c8-(5).pt")

