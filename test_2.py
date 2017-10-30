#coding:utf-8
f = open("3000+class")
char_to_label = {}
tmp_line = f.readline()
while tmp_line != "":
    tmp_char = tmp_line.split(" ")[0]
    tmp_label = tmp_line.split(" ")[1][:-1]
    char_to_label[unicode(tmp_char,"utf-8")] = tmp_label
    tmp_line = f.readline()
print char_to_label
print char_to_label[u"人"]
tmp_line = u"人"
tmp_line.decode("utf-8")