import sys
nums = []

vals = {}

for l in open(sys.argv[1]).readlines():
    parts = l.strip().split()
    if len(parts) == 4:
        x, y, v, s = parts[0], parts[1], parts[2], parts[3]

        v = str(round(float(v), 2))

        value = "$" + v + "$"
        if float(v) < 0.1:
            value = '\\textbf{' + str(v) + '}'
        elif float(v) > 0:
            value = value + "\\newline {\\tiny $ \pm" + s + "$}"


        vals[(x,y)] = value
        # print(x)
        # print(y)
        # print(value)

# n1 = [3, 4, 5, 7, 6, 1, 2]
# n2 = [6, 9, 8, 10, 3, 4, 5, 7, 13, 11, 12, 1, 2]
# n2 = list(range(1, 14))
n1 = ["AV", "DYN", "DYN2", "COV", "COVC", "AVMIN", "AVC",]
n2 = ["NO", "MASS", "TAN", "M1", "TR",
      "CR1", "CR2", "CUR", "ACOR", "M1.5", "M2", ]
# n1 = ["AV", "DYN", "DYN2", "COV", "COVC", "AVMIN", "AVC",]
# n2 = ["NO"]

print('\\begin{tabular}{' + ('>{\centering}m{0.04\linewidth}<{\centering}'*(len(n2) + 1)) + '}')
print(" & ".join(["-"] + [str(i) for i in n2]) + "\\tabularnewline\\hline")
for i in n1:
    s = "{i}".format(i=i)
    for j in n2:
        v = vals.get((str(i), str(j)), "-")
        # s += "{:.3}\t".format(v)
        s += " & {}".format(v)
    s += "\\tabularnewline\\hline"
    print(s)
print('\end{tabular}')

