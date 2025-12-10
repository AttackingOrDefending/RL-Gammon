class Piece:

    def __hash__(self):
        return hash(self.__class__.__name__)
    def __eq__(self,other):
        return self.__class__.__name__ == other.__class__.__name__

    def possibles_promotions(self):
        print(self.__class__.__name__)
        raise NotImplemented

    def prises(self, i, j):
        raise NotImplemented

    def mouvements(self, i, j):
        raise NotImplemented

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__