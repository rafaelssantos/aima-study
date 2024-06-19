from search.search import *

eight_puzzle = EightPuzzle((1, 2, 3, 4, 5, 7, 8, 6, 0))

print (astar_search(eight_puzzle).solution())