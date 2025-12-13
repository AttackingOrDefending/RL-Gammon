from discord_interface.player.model.textual_ai import Textual_AI
from discord_interface.utils.mytime import Time


class GTP_AI_Go(Textual_AI):

    async def replay_history(self):

        for color, move in self.history:
            await self.send('play ' + color + ' ' + move)

    async def reset(self):

        await super().reset()

        self.game.reset()

        await self.send('clear_board')


    async def undo(self):
        await self.send('undo')
        self.game.undo()
        self.history.pop()


    def get_response_symbol(self):
        return '='

    def get_error_symbol(self):
        return '?'


    async def invalid_action_processing(self):
        await self.undo()

    async def replays(self, action):


        self.game.plays(action)

        action = self.action_to_string(action)

        color = self.self_color()
        move = self.move_conversion_to_gtp(action)
        await self.send('play '+color+' '+move)
        self.history.append((color, move))



    async def opponent_plays(self, action):
        self.game.plays(action)

        action = self.action_to_string(action)

        color = self.opponent_color()
        move = self.move_conversion_to_gtp(action)
        await self.send('play '+ color +' '+move)
        self.history.append((color, move))




    async def plays(self, time_left: Time=None, opponent_time_left: Time=None):
        #print('A')
        await self.send('time_left ' + self.self_color() + ' ' + str(time_left.to_seconds()))
        await self.send('time_left ' + self.opponent_color() + ' ' + str(opponent_time_left.to_seconds()))
        #print('A')

        action = await self.send('genmove '+self.self_color())
        self.history.append((self.self_color(), action))

        #print("'"+action+"'")
        action = self.move_conversion_from_gtp(action)
        #print("'"+self.move_conversion_from_gtp(action)+"'")


        action = self.string_to_action(action)

        self.game.plays(action)

        return action

    def move_conversion_from_gtp(self, move):
        action=''
        is_number=False
        if '-' not in move:
            for c in move:
                if is_number and not c.isdigit():
                    action += '-'+c
                else:
                    action += c
                if c.isdigit():
                    is_number= True
        #print('>',move, '>>', action)
            if action[-1].isdigit():
                return action.upper()
        return action

    def move_conversion_to_gtp(self, action):
        return action.lower().replace('-','')

