import asyncio
from discord_interface.player.model.player import *



import traceback


from re import compile

from discord_interface.utils.pattern_enum import EnumPattern
from discord_interface.utils.terminal import red

correspondance_hex = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's'}

anti_correspondance_hex = {'c': 2, 'h': 7, 's': 18, 'j': 9, 'e': 4, 'f': 5, 'o': 14, 'b': 1, 'r': 17, 'k': 10, 'i': 8, 'g': 6, 'l': 11, 'n': 13, 'q': 16, 'p': 15, 'd': 3, 'a': 0, 'm': 12}



class Textual_AI(Player):


    def __init__(self, program_name, program_arguments='', program_directory='', game:Game = None, search_time_per_action=1, logfile=None, history_file='history.json', timeout=60*30, time_for_program_quit=10, move_keywords=[], endgame_imply_quit=True):#advance_profiling=False, back_first=True,

        if logfile is None:
            self.logfile_name = program_name+'_gtp.log'
        else:
            self.logfile_name = logfile

        self.last_command = ''

        if program_arguments is None:
            program_arguments = ''

        if program_directory is None:
            program_directory = ''

        try:
            #print('* init')

            self.search_time_per_action = search_time_per_action
            self.time_for_program_quit=time_for_program_quit

            self.timeout = timeout
            self.history = []

            self.move_keywords_init = move_keywords

            if program_directory != '' and program_directory[-1] != '/':
                program_directory=program_directory+'/'

            self.program_directory = program_directory

            self.cmd = program_directory + program_name+' '+program_arguments

            self.history_file=program_directory + history_file

            #self.advance_profiling = advance_profiling
            self.black_first = True


            self.endgame_imply_quit = endgame_imply_quit

            super().__init__(game=game)

            self.init()

        except Exception as e:
            traceback.print_exc()
            raise e

    def init(self):
        import pexpect

        self.logfile = open(self.logfile_name, 'w')#self.program_directory

        self.processus = pexpect.spawn(self.cmd, encoding='utf-8', codec_errors='replace', logfile=self.logfile)

    async def reset(self) -> None:
        super().__init__(game=self.game)
        self.history = []

    async def update_game(self, game_name:str):
        super().update_game(game_name)
        #self.game = None
        #self.game_name = game_name

        game = EnumGames.find_game(game_name)
        if game is None:
            game_move_keywords = []
        else:
            game_move_keywords = game.move_keywords

        self.move_keywords = self.move_keywords_init + game_move_keywords
        self.build_move_verifier()

        if self.processus is None:
            self.init()

        await self.send('game '+game_name)

    async def replay_the_match(self):
        await self.terminate()

        self.init()

        await self.send('game '+self.game_name)

        await self.replay_history()


    def get_white_words(self):
        return ['w', 'white']

    def get_black_words(self):
        return ['b', 'black']

    def get_chance_words(self):
        return ['c', 'chance']

    def build_move_verifier(self):

        if self.move_keywords == []:
            move_pattern = EnumPattern.RAW_PATTERN_MOVE.value
        else:
            move_pattern = EnumPattern.RAW_PATTERN_MOVE.value + "|" + "|".join(self.move_keywords)
        new_global_pattern = "(" + move_pattern + ")" + "(-(" + move_pattern + "))*"

        self.move_verifier = compile(new_global_pattern)

    def self_color(self):
        if self.player_number == 0 and self.black_first or self.player_number == 1 and not self.black_first:
            return 'b'
        else:
            assert self.player_number == 0 and not self.black_first or self.player_number == 1 and self.black_first
            return 'w'

    def opponent_color(self):
        if self.player_number == 0 and self.black_first or self.player_number == 1 and not self.black_first:
            return 'w'
        else:
            assert self.player_number == 0 and not self.black_first or self.player_number == 1 and self.black_first
            return 'b'

    async def get_current_player(self) -> int:
        """if not self.advance_profiling:
            return 0
        else:"""
        player = (await self.send('player')).lower()

        if player in self.get_black_words() and self.black_first or player in self.get_white_words() and not self.black_first:
            return 0

        elif player in self.get_white_words() and self.black_first or player in self.get_black_words() and not self.black_first:
            return 1

        elif player.isdigit():
            return int(player)

        elif player in self.get_chance_words():
            return -1

        else:
            return 0


    async def send(self, line):
        await asyncio.to_thread(self.processus.sendline, line)
        self.last_command=line
        return await self.recv()


    async def end(self):
        if self.endgame_imply_quit:
            await self.terminate()
        await super().end()

        self.processus = None



    async def terminate(self):

        await asyncio.to_thread(self.processus.sendline, 'quit') # ne pas utiliser send !

        await asyncio.sleep(self.time_for_program_quit)

        await asyncio.to_thread(self.processus.terminate)
        await asyncio.to_thread(self.processus.wait)
        await asyncio.to_thread(self.processus.close)

        await asyncio.to_thread(self.logfile.close)

    def get_response_symbol(self):
        return ''

    def get_error_symbol(self):
        return ''

    async def recv(self):

        if self.get_error_symbol():

            await asyncio.to_thread(self.processus.expect, r'\r\n(=|\?)', timeout=self.timeout)
            #match = self.processus.match.group(1)

            #if '=' in match:
            if '=' in self.processus.after:
                i = 0
            else:
                assert '?' in self.processus.after
                i = 1

        else:
            await asyncio.to_thread(self.processus.expect, '\r\n'+self.get_response_symbol(), timeout=self.timeout)
            i = 0

        await asyncio.to_thread(self.processus.expect, '\r\n', timeout=self.timeout)

        response = self.processus.before


        if i == 1:
            print(red('GTP Failure Response!'))
            print('GTP response: ', '?'+response)
            print('Last command: ',self.last_command)

        if '#' in response:

            while ' #' in response:
                response = response.replace(' #','#')

            comments = self.processus.before.split('#')[1:]
            response = response.split('#')[0]

            for comment in comments:
                print(comment)

        if i == 1:
            raise Exception('GTP Failure Response!')

        return response.strip()


