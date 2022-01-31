import sys
import pstats
from pstats import SortKey
fname = sys.argv[1]
p = pstats.Stats(fname)
p.sort_stats(SortKey.TIME).print_stats(400)
