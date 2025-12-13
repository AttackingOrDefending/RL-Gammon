first_move = True
move = '5-6'


move = move.replace('-', ' ')
rolls = move.split()
if first_move:
    with open('first_moves.txt') as f:
        data = f.read()
else:
    with open('moves.txt') as f:
        data = f.read()
lines = data.split('\n')
for line in lines:
    roll = line[line.index('l: ') + 3:line.index(')')]
    if roll == ''.join(rolls) or roll == ''.join(rolls[::-1]):
        if not first_move or ('X starts' in line and int(rolls[0]) > int(rolls[1])) or ('O starts' in line and int(rolls[1]) > int(rolls[0])):
            print(line.split()[-1])
