from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys

# inputprefix = "out_sc0_GK_baseline"
inputprefix = sys.argv[1]
inputgrid = inputprefix + ".fits.gz"
print(inputgrid)

hdulist = fits.open(inputgrid)
cumulative_array = hdulist[0].data
kiclist = np.asarray(hdulist[1].data, dtype=np.int32)
probdet = cumulative_array[0]
probtot = cumulative_array[1]
prihdr = hdulist[0].header
min_period = prihdr["MINPER"]
max_period = prihdr["MAXPER"]
n_period = prihdr["NPER"]
min_rp = prihdr["MINRP"]
max_rp = prihdr["MAXRP"]
n_rp = prihdr["NRP"]
print ("KIC list length" + '{:6d}'.format(kiclist.size))


period_want = np.linspace(min_period, max_period, n_period)
rp_want = np.linspace(min_rp, max_rp, n_rp)
period_want2d, rp_want2d = np.meshgrid(period_want, rp_want)

mynearblack = tuple(np.array([75.0, 75.0, 75.0]) / 255.0)
myblue = tuple(np.array([0.0, 109.0, 219.0]) / 255.0)
labelfontsize=9.0
tickfontsize=14.0
datalinewidth=3.0
plotboxlinewidth=3.0

wantFigure = inputprefix
plt.figure(figsize=(6,4.5),dpi=300)
ax = plt.gca()
ax.set_position([0.125, 0.125, 0.825, 0.825])
uselevels = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

X = period_want2d
Y = rp_want2d
x1 = probdet / kiclist.size
CS2 = plt.contour(X, Y, x1, levels=uselevels, linewidth=datalinewidth, 
                   colors=(myblue,) * len(uselevels))
plt.clabel(CS2, inline=1, fontsize=labelfontsize, fmt='%1.2f',
           inline_spacing=6.0)
CS1 = plt.contourf(X, Y, x1, levels=uselevels, cmap=plt.cm.bone)    
plt.xlabel('Period [day]', fontsize=labelfontsize, fontweight='heavy')
plt.ylabel('R$_{p}$ [R$_{\oplus}$]', fontsize=labelfontsize, 
            fontweight='heavy')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(plotboxlinewidth)
    ax.spines[axis].set_color(mynearblack)
ax.tick_params('both', labelsize=tickfontsize, width=plotboxlinewidth, 
               color=mynearblack, length=plotboxlinewidth*3)
plt.savefig(wantFigure+'.png',bbox_inches='tight')
plt.savefig(wantFigure+'.eps',bbox_inches='tight')
