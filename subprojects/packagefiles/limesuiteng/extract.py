import os
import sys
import zipfile

names = {"kiss_fft.c", "kiss_fft.h", "_kiss_fft_guts.h", "kiss_fft_log.h", "rang.hpp", "INI.h"}
out = sys.argv[1]
for archive in sys.argv[2:]:
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name in names and member.count("/") <= 2 and "/test" not in member:
                with open(os.path.join(out, name), "wb") as f:
                    f.write(z.read(member))
