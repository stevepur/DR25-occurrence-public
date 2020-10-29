import glob
import gzip
import os
from astropy.io import fits
import numpy as np

# prefix = "detCompOut"
prefix = "out__"
filelist = glob.glob(prefix + '[0-9][0-9][0-9][0-9].fits.gz')

# open first fits file to get the structure
hdulist = fits.open(filelist[0])
prihdr = hdulist[0].header
cumulative_array = hdulist[0].data
print(cumulative_array.shape)
cumulative_kiclist = np.asarray(hdulist[1].data, dtype=np.int32)
hdulist.close()
nstars = np.zeros(prihdr["NTEFF"])
for s in range(prihdr["NTEFF"]):
    nstars[s] = prihdr["NSTEFF" + str(s)]
print(nstars)
#len(filelist)
for i in range(1,len(filelist)):
    print (i)
    hdulist = fits.open(filelist[i])
    cumulative_array = cumulative_array + hdulist[0].data
    print(cumulative_array.shape)
    print("length of kiclist: " + str(len(hdulist[1].data)))
    cumulative_kiclist = np.append(cumulative_kiclist, 
                           np.asarray(hdulist[1].data, dtype=np.int32))
    sprihdr = hdulist[0].header
    for s in range(sprihdr["NTEFF"]):
        nstars[s] = nstars[s] + sprihdr["NSTEFF" + str(s)]
    print(nstars)
    hdulist.close()

output_filename = prefix + '.fits'
# Package data into fits file
print(cumulative_array.shape)
hdu = fits.PrimaryHDU(cumulative_array)
hdulist = fits.HDUList([hdu])
newprihdr = hdulist[0].header
for i in range(prihdr["NTEFF"]):
    prihdr.set("NSTEFF" + str(i),nstars[i])
newprihdr.extend(prihdr.copy(strip=True))
newcol = fits.Column(name='kiclist', format='J', array=cumulative_kiclist)
cols = fits.ColDefs([newcol])
tbhdu = fits.BinTableHDU.from_columns(cols)
hdulist.append(tbhdu)
hdulist.writeto(output_filename, clobber=True)
f_in = open(output_filename, 'rb')
f_out = gzip.open(output_filename + '.gz', 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()
os.remove(output_filename)
