from discord_interface.games.instances.amazons import AmazonsDiscord


class AmazonsDiscordBugger(AmazonsDiscord):

    def action_to_string(self, object):
        i, j = object

        if isinstance(i, tuple) and isinstance(j, tuple):
            i1, j1 = i
            i2, j2 = j

            return self.CORRESPONDENCE[j1] + str(i1 + 1) + '-' + self.CORRESPONDENCE[j2] + str(i2 + 1)
        else:
            self.CORRESPONDENCE[j] + str(1 + 1)