import os
print ord("a"), ord("A")
tmp_line = ""
tmp_seq = ""
for i in range(10):
    tmp_line += "\'{0}\':{1},".format(i,i)
    tmp_seq += "{0}".format(i)
for i in range(ord("a"), ord("a")+26):
    tmp_line += "\'{0}\':{1},".format(chr(i),i-ord("a")+10)
    tmp_seq += "{0}".format(chr(i))
for i in range(ord("A"),ord("A")+26):
    tmp_line += "\'{0}\':{1},".format(chr(i),i-ord("A")+36)
    tmp_seq += "{0}".format(chr(i))
print tmp_line
print tmp_seq