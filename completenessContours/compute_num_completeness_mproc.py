import os
import argparse
import gzip
import numpy as np
import pandas as pd
from astropy.io import fits
import compute_num_completeness_w_ve as kp

def read_stellar_table(filename):
    kic = pd.read_csv(filename)
    print("length of kic IDs: " + str(len(kic.kepid)))
    print("length of unique kic IDs: " + str(len(np.unique(kic.kepid))))
    all_datas = []
    all_keys = []
    for i in range(len(kic.kepid)):
        cur = kp.kepler_single_comp_data()
        cur.id = kic.kepid[i]
        cur.rstar = kic.radius[i]
        cur.logg = kic.logg[i]
        cur.teff = kic.teff[i]
        cur.dataspan = kic.dataspan[i]
        cur.dutycycle = kic.dutycycle[i]
        cur.limbcoeffs = np.array([kic.limbdark_coeff1[i], kic.limbdark_coeff2[i], kic.limbdark_coeff3[i], kic.limbdark_coeff4[i]])
        cur.cdppSlopeLong = kic.cdppslplong[i]
        cur.cdppSlopeShort = kic.cdppslpshrt[i]
        all_datas.append(cur)
        all_keys.append(cur.id)
    # Now zip it all together into dictionary with kic as key
    stellar_dict = dict(zip(all_keys, all_datas))
    print("length of stellar_dict: " + str(len(stellar_dict)))
    return stellar_dict, kic.kepid

def nas_multi_grid_dr25(worker_id, n_workers, min_period, max_period,
                         n_period, min_rp, max_rp, n_rp, 
                         stellar_database,
                        planet_metric_path, ve_data_file,
                        ve_model_name, output_prefix):

    print("worker id " + str(worker_id))
    # Define the grids and data parameters
    period_want = np.linspace(min_period, max_period, n_period)
    rp_want = np.linspace(min_rp, max_rp, n_rp)
    # Load the stellar data 
    stellar_database_filename = stellar_database
    stellar_dict, kiclist = read_stellar_table(stellar_database_filename)
    print("stellar_dict length: " + str(len(stellar_dict)))
    # kiclist = [cur.id for cur in stellar_dict.values()]
    print("kiclist length: " + str(len(kiclist)))

    # Ensure kic list is sorted
    kiclist = np.sort(kiclist)
    print("sorted kiclist length: " + str(len(kiclist)))
    
    period_want2d, rp_want2d = np.meshgrid(period_want, rp_want)
    shape2d = period_want2d.shape
    shape3d = (2,) + shape2d
    # Create zero arrays that we will fill
    cumulative_array = np.zeros(shape3d, dtype=np.float64)
    usekiclist = np.array([], dtype=np.int32)
    doneOnce = False
    for i in range(len(kiclist)):
        if (i%1000 == 0):
            print(str(np.round(100.*np.double(i)/len(kiclist), 2)) + "%")
        if np.mod(i, n_workers) == worker_id :
            curid = kiclist[i]
            windowfunc_filename = os.path.join(planet_metric_path,'kplr' + \
                                  '{:09d}'.format(curid) + \
                                  '_dr25_window.fits')
            if not os.path.isfile(windowfunc_filename):
                print("worker " + str(worker_id) + " skipping " + windowfunc_filename)
                continue
            usekiclist = np.append(usekiclist, curid)
            curdict = stellar_dict[curid]
            curdict.period_want = period_want
            curdict.rp_want = rp_want
            curdict.ecc = 0.0
            curdict.planet_detection_metric_path = planet_metric_path
            curdict.ve_fit_filename = ve_data_file
            curdict.ve_model_name = ve_model_name
            if not doneOnce:
                DEMod = None
            probdet, probtot, DEMod = kp.kepler_single_comp_dr25(curdict, DEMod=DEMod)
            doneOnce = True

            if np.logical_not(np.all(np.isfinite(probdet))):
                print ("non finite probdet detected.  skipping.")
                print (curid)
                continue
            if np.logical_not(np.all(np.isfinite(probtot))):
                print ("non finite probtot detected.  skipping.")
                print (curid)
                continue
            cumulative_array[0] = cumulative_array[0] + probdet
            cumulative_array[1] = cumulative_array[1] + probtot
    # check that the final cumulative_array is clean
    if np.logical_not(np.all(np.isfinite(cumulative_array))):
        print ("Non finite cumulative_array Found!")
        print (worker_id)
    output_filename = output_prefix + "_" + '_{:04d}'.format(worker_id) + '.fits'
    # Package data into fits file
    # Add cumulative_array to primary fits image data
    hdu = fits.PrimaryHDU(cumulative_array)
    hdulist = fits.HDUList([hdu])
    # Add parameters to primary header
    prihdr = hdulist[0].header
    prihdr.set('MINPER',min_period)
    prihdr.set('MAXPER',max_period)
    prihdr.set('NPER',n_period)
    prihdr.set('MINRP',min_rp)
    prihdr.set('MAXRP',max_rp)
    prihdr.set('NRP',n_rp)
    prihdr.set('STELDATA',stellar_database_filename)
    prihdr.set('DEFILE',ve_data_file)
    prihdr.set('DEMODEL',ve_model_name)
    # Now add the kics used in this chunk of the data
    newcol = fits.Column(name='kiclist', format='J', array=usekiclist)
    cols = fits.ColDefs([newcol])
    #    tbhdu = fits.new_table(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    hdulist.append(tbhdu)
    # Write out fits file
    hdulist.writeto(output_filename, clobber=True)
    hdulist.close()
    # Gzip fits file
    f_in = open(output_filename, 'rb')
    f_out = gzip.open(output_filename + '.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(output_filename)

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_id", type=int,
                        help="Unique worker ID number")
    parser.add_argument("n_workers", type=int,
                        help="Total number of workers")
    parser.add_argument("min_period", type=float,
                        help="Orbital period lower limit")
    parser.add_argument("max_period", type=float,
                        help="Orbital period upper limit")
    parser.add_argument("n_period", type=int,
                        help="Number of steps in grid over orbital period")
    parser.add_argument("min_rp", type=float,
                        help="Planet radius lower limit")
    parser.add_argument("max_rp", type=float,
                        help="Planet radius upper limit")
    parser.add_argument("n_rp", type=int,
                        help="Number of steps in grid over planet radius")
    parser.add_argument("stellar_database", type=str,
                        help="name of stellar table filename")
    parser.add_argument("planet_metric_path", type=str,
                        help="Path to the planet detection metric files")
    parser.add_argument("ve_data_file", type=str,
                        help="filename of the vetting efficiency data file")
    parser.add_argument("ve_model_name", type=str,
                        help="name of the vetting efficiency model")
    parser.add_argument("output_prefix", type=str,
                        help="Prefix for output files")
    args = parser.parse_args()
    outputresult = nas_multi_grid_dr25(args.worker_id, args.n_workers,
                                        args.min_period, args.max_period, args.n_period,
                                        args.min_rp, args.max_rp, args.n_rp,
                                        args.stellar_database,
                                        args.planet_metric_path,
                                        args.ve_data_file,
                                        args.ve_model_name,
                                        args.output_prefix)
