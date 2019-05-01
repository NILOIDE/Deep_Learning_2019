import codecs

if __name__ == "__main__":
    abc = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ò',
           ',', '.', ':', '?', '!', ';', '\'', '\n', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö'}
    # nonascii = bytearray(range(0x80, 0x100))
    with codecs.open("Viimeinen_syksy.txt", 'r', 'utf-8') as infile, codecs.open("Viimeinen_syksy_clean.txt", 'w', 'utf-8') as outfile:
        for line in infile:  # b'\n'-separated lines (Linux, OSX, Windows)
            for c in line:
                print(c)
                outfile.write(c if c in abc else '')
                # outfile.write(line.translate(str.maketrans(nonascii, '')))
            print(line)
            breakpoint()

