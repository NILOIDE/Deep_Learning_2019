import codecs

if __name__ == "__main__":
    abc = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'å', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ò', '(', ')',
           ',', '.', ':', '?', '!', ';', '-', '\'', '\n', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö'}
    # abc = ''.join(abc)
    # abc = abc.encode('utf-8')
    # print(abc)# I have also tried without this
    with codecs.open("Viimeinen_syksy.txt", 'r', 'latin1', errors='ignore') as infile, codecs.open("Viimeinen_syksy_clean.txt", 'w') as outfile:
        for line in infile:
            for c in line:
                if c in abc:
                    outfile.write(c)
