#
# HRK 2022 
#
# Determine the sky area coverage of mutliple levels and RMA files. 
#


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from shapely.geometry import Polygon
from astropy.io import fits
import numpy.ma as ma
from astropy.wcs import wcs
from skimage import measure
from copy import copy,deepcopy
from collections import OrderedDict


def cal_imagepixval(axis,pixel,header):
    """
    Calculates the correct value of the pixel
    according to email exchange with Eric Geisen 
    freq = freq0 + (ch - refch) * increment
    """
    # get the index of the coordinates
    idx = 0
    if header['ctype1'] == axis:
        idx = 1

    if header['ctype2'] == axis:
        idx = 2

    value = header['crval'+str(idx)] + (pixel - header['crpix'+str(idx)]) * header['cdelt'+str(idx)]

    return value
    


def filenamecounter(fname,extention='.png'):
    """
    just provide a filename taken exsisting files with the
    same name in the directory into accout
    """
    import os

    filename = fname+extention

    if os.path.isfile(fname+extention):

        counter = 0
        filename = fname+'_'+'{}'+extention
        while os.path.isfile(filename.format(counter)):
            counter += 1
        filename = filename.format(counter)

    return filename


def ellipse_RA_check(radec):
    """
    Split the polygons into sub-polygons to be checked
    """
    new_polygons = []


    # check sources if they go over 360 degrees
    ra_check       = abs(np.diff(radec[:,0])) > 300
    ra_check_where = np.where(ra_check)[0].flatten()

    selit          = np.zeros(len(radec)).astype('bool')


    if len(ra_check_where)%2 == 0:

        for i in range(int(len(ra_check_where)/2)):            
            selit[ra_check_where[2*i]+1:ra_check_where[2*i+1]+1] = True


        radec_list_1 = radec[selit].tolist()

        if radec_list_1[0] == radec_list_1[-1]:
                new_polygons.append(radec_list_1)
        else:
                radec_list_1.append(radec_list_1[0])
                new_polygons.append(radec_list_1)


        radec_list_2 = radec[np.invert(selit)].tolist()

        if radec_list_2[0] == radec_list_2[-1]:
                new_polygons.append(radec_list_2)
        else:
                radec_list_2.append(radec_list_2[0])
                new_polygons.append(radec_list_2)


    elif len(ra_check_where) == 1:
        print('strange polygon 1',radec)
        sys.exit(-1)

        # ellipse crosses once the RA border
        selit[ra_check_where[0]+1] = False
        new_polygons.append(radec[selit].tolist())
    elif len(ra_check_where) == 0:
        print('strange polygon 0',radec)
        sys.exit(-1)

        # ellipse crosses never the RA border
        new_polygons.append(radec)
    else:
        print('strange polygon',radec)
        sys.exit(-1)


    return new_polygons




def polygon_area_on_sphere(lats, lons, algorithm = 0, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified. Otherwise, in the units of provided radius.
    lats and lons are in degrees.

    https://stackoverflow.com/questions/1340223/calculating-area-enclosed-by-arbitrary-polygon-on-earths-surface

    HRK: changed the output to square degrees if radius is None

    """
    #TODO: take into account geodesy (i.e. convert latitude to authalic sphere, use radius of authalic sphere instead of mean radius of spherical earth)

    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)

    if algorithm==0:
        # Line integral based on Green's Theorem, assumes spherical Earth
        from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff

        #close polygon
        if lats[0]!=lats[-1]:
            lats = append(lats, lats[0])
            lons = append(lons, lons[0])

        # Get colatitude (a measure of surface distance as an angle)
        a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
        colat = 2*arctan2( sqrt(a), sqrt(1-a) )

        #azimuth of each point in segment from the arbitrary origin
        az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)

        # Calculate step sizes
        # daz = diff(az) % (2*pi)
        daz = diff(az)
        daz = (daz + pi) % (2 * pi) - pi

        # Determine average surface distance for each step
        deltas=diff(colat)/2
        colat=colat[0:-1]+deltas

        # Integral over azimuth is 1-cos(colatitudes)
        integrands = (1-cos(colat)) * daz

        # Integrate and save the answer as a fraction of the unit sphere.
        # Note that the sum of the integrands will include a factor of 4pi.
        area = abs(sum(integrands))/(4*pi) # Could be area of inside or outside

        area = min(area,1-area)
        if radius is not None: #return in units of radius
            return area * 4*pi*radius**2
        else: #return in ratio of sphere total area
            return area * 360**2 / np.pi



def test_contour():
    """
    just for me to play
    """
    a = np.zeros((10,30))


    a[5][15] = 1
    a[6][15] = 1

    contours = measure.find_contours(a)

    print(contours)

    fig, ax = plt.subplots()
    ax.imshow(a, cmap=plt.cm.gray,origin="lower")

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()

    sys.exit(-1)


def determine_FoV_verticees(image):
    """
    determine the verticees of the FoV 
    """
    
    #
    # Determine the FoV of the image
    #

    # perpare the image into binary image
    get_fov_image            = copy(image)
    sel_image                = image > 0
    get_fov_image[sel_image] = 1
    get_fov_image_no_nan     = np.nan_to_num(get_fov_image,nan=0)
    #

    # countours are counter-clockwise and are in pixel coordinates (y,x)
    # 
    contours = measure.find_contours(get_fov_image_no_nan,0.1,fully_connected='low',positive_orientation='low')

    # for the FoV case merge all contours into one (they are split due to the image borders)
    #
    im_y  = []
    im_x  = []
    for i in range(len(contours)):
            if len(contours[i][:, 1]) < 3:
                print('FoV Contours have only 2 verticees')
            im_y += list(contours[i][:, 0])
            im_x += list(contours[i][:, 1])

    FoV_verticees = np.array([im_y,im_x]).T

    # Check if the contour is closed
    # most likely the contour is broken ue dto image 
    # borders
    #
    if check_contour_close(FoV_verticees)[0]== False:
        
        # re-build the FoV_verticees
        new_contour = []

        im_shape = np.shape(get_fov_image_no_nan)

        # check FoV boundary is broken 
        # assume rectangular image
        #
        bounrary_y0_x   = get_fov_image_no_nan[0,:]
        bounrary_ymax_x = get_fov_image_no_nan[-1,:]
        bounrary_y_x0   = get_fov_image_no_nan[:,0]   # to get it counter clockwise
        bounrary_y_xmax = get_fov_image_no_nan[:,-1]  # to get it counter clockwise

        b1_idx_x  = np.ravel(np.where(bounrary_y0_x > 0))
        b2_idx_x  = np.ravel(np.where(bounrary_ymax_x > 0))[::-1]
        b3_idx_y  = np.ravel(np.where(bounrary_y_x0 > 0))[::-1]
        b4_idx_y  = np.ravel(np.where(bounrary_y_xmax > 0))

        b1_idx_y = np.zeros(len(b1_idx_x))
        b2_idx_y = np.ones(len(b2_idx_x)) * im_shape[0]
        b3_idx_x = np.zeros(len(b3_idx_y))
        b4_idx_x = np.ones(len(b4_idx_y)) * im_shape[1]


        concatenatex = [b1_idx_x,b4_idx_x,b2_idx_x,b3_idx_x]  # get it counter clockwise
        concatenatey = [b1_idx_y,b4_idx_y,b2_idx_y,b3_idx_y]

        # use only stuff with length > 0
        new_x = []
        new_y = []
        for k in range(len(concatenatex)):
            if len(concatenatex[k]) > 0:
                new_x.append(concatenatex[k])
                new_y.append(concatenatey[k])

        if len(contours) != len(new_y):
            print('bugger !!! check FoV verticees')
            sys.exit(-1)

        for c in range(len(contours)):
                new_contour.extend(contours[c])            
                # do the add on
                add_on_b = np.array([new_y[c][:],new_x[c][:]]).T
                new_contour.extend(add_on_b)

        FoV_verticees = np.array(new_contour)
        
        if check_contour_close(FoV_verticees)[0]== False:
            print('Something with the broken FoV verticees and the fixing went wrong ')
            sys.exit(-1)

    cont_ra,cont_dec = np.array(FoV_verticees).T

    #fig, ax = plt.subplots()
    #ax.plot(cont_ra,cont_dec,color='blue')
    #plt.show()

    return FoV_verticees



def check_stats_on_verticees(image,verticees):
    """
    estimate the minimum and maximum image values of on the verticees

    """

    fov_x = np.round(verticees[:,0]).astype('int')
    fov_y = np.round(verticees[:,1]).astype('int')

    sel_x = fov_x >= image.shape[0]
    sel_y = fov_y >= image.shape[1]

    fov_x[sel_x]   = image.shape[0] -1
    fov_y[sel_y]   = image.shape[1] -1

    r_mask1        = np.zeros(image.shape, dtype='bool')
    r_mask1[fov_x,fov_y] = 1

    # Estimate the min on the contour and use this to discriminate if the FoV contour needs to be taken into account in the 
    # following steps
    # please note the strange masked_array setting that 1 are bad, so ~ is invert the mask
    #
    FoV_border_min = np.nanmin(ma.masked_array(image,~r_mask1))
    FoV_border_max = np.nanmax(ma.masked_array(image,~r_mask1))
    
    return FoV_border_min, FoV_border_max


def check_contour_close(contour):
    """
    """
    sigma = 5
    
    # strange that does not work print(np.logical_not(np.linalg.norm(contour[-1] - contour[0]) == 0))
    diff = []
    for k in range(len(contour)-1):
        diff.append(np.linalg.norm(contour[k+1] - contour[k]))

    # print( np.linalg.norm(contour[-1] - contour[0]), np.median(diff), sigma * np.std(diff))
    # print([contour[0],contour[-1]])

    if np.linalg.norm(contour[-1] - contour[0]) < np.median(diff) + sigma * np.std(diff):
        close =  True
    else:
        close = False

    if np.linalg.norm(contour[-1] - contour[0]) < 3:
        close = True

    return close, [contour[0],contour[-1]]



def cut_out_fov_polygon(contour,FoV_verticees):
    """
    cut out the polygon out of the FoV
    """

    # determine the closest point of the FoV verticees 
    # to the contour start and end
    #
    distance_p1     = []
    distance_p2     = []
    for cf in range(len(FoV_verticees)):
        distance_p1.append(np.linalg.norm(FoV_verticees[cf] - contour[0]))
        distance_p2.append(np.linalg.norm(FoV_verticees[cf] - contour[-1]))
    #
    idx_p1 = np.argmin(distance_p1) 
    idx_p2 = np.argmin(distance_p2)
        

    # determine the low and upper index
    if idx_p1 < idx_p2:
        idx_low = idx_p1
        idx_up  = idx_p2
    else:
        idx_low = idx_p2
        idx_up  = idx_p1
     
    # check for the shortest cut out 
    length_1 = idx_up - idx_low
    length_2 = idx_low - 0 + abs(len(FoV_verticees) - idx_up)

    # cut it out
    cut_out = []
    if length_1 < length_2:
        cut_out.extend(FoV_verticees[idx_low:idx_up]) 
    else:
        cut_out.extend(FoV_verticees[idx_up:])
        cut_out.extend(FoV_verticees[:idx_low])


    return np.array(cut_out)


def close_polygon_with_fov(contour,FoV_verticees,bounds):
    """
    """

    new_contour = contour.tolist()

    # just get the cut out
    fov_cut_out = cut_out_fov_polygon(contour,FoV_verticees)

    cut_ra , cut_dec  = np.array(fov_cut_out).T

    #fig, ax = plt.subplots()
    #ax.scatter(cut_ra,cut_dec,color='red')
    #ax.text(cut_ra[0],cut_dec[0],'CLOSE Cut Start')
    #ax.text(cut_ra[-1],cut_dec[-1],'CLOSE Cut End')
    #ax.text(contour[-1][0],contour[-1][0],'CLOSE Cont Start')
    #ax.text(contour[-1][-1],contour[-1][-1],'CLOSE Cont End')
    #plt.show()
    #print('contour End',contour[-1],' add on start',fov_cut_out[0])
    #print('contour start',contour[0],'add on ends',fov_cut_out[-1])
    
    # check how it connects
    #
    if np.linalg.norm(fov_cut_out[0] - contour[-1]) < np.linalg.norm(fov_cut_out[-1] - contour[-1]):
        new_contour.extend(fov_cut_out)
    else:
        new_contour.extend(fov_cut_out[::-1])


    cut_ra , cut_dec  = np.array(new_contour).T


    percentage_of_closeing_polygon      = len(fov_cut_out)/len(contour)
    percentage_of_closeing_polygon_FoV  = len(fov_cut_out)/len(FoV_verticees)


    return new_contour, percentage_of_closeing_polygon, percentage_of_closeing_polygon_FoV




#
# File name
#


def area_coverage(fits_file_names,levels,doplot_pixel_image,pltsave_pixel_image,doplot_world_contours,pltsave_world_contours,\
                                              print_total_area_info,percentage_area,do_plot_cont_labels,DPI):
    """
    try to estimate the areac coverage in an 
    image
    """

    # init the estimates
    #
    tot_area          = np.zeros(len(levels))
    tot_area_levels   = np.zeros(len(levels))
    #
    contours_to_check = []
    #
    full_info                   = {}
    full_info_contours_to_check = {}
    #
    for fn in range(len(fits_file_names)):

        # set the info
        full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')] = {}
        full_info_contours_to_check[fits_file_names[fn].replace('.fits','').replace('.FITS','')] = {}

        # Get FITS data
        sky_image = fits.getdata(fits_file_names[fn])
        # get header info
        opft = fits.open(fits_file_names[fn])
        imheader = opft[0].header
        opft.close()

        # optain the header info
        ow = wcs.WCS(imheader)

        # define a 2 dim header to get the calulation done
        w = wcs.WCS(naxis=2)
        w.wcs.crval     = [ow.wcs.crval[0],ow.wcs.crval[1]]
        w.wcs.crpix     = [ow.wcs.crpix[0],ow.wcs.crpix[1]]
        w.wcs.cdelt     = [ow.wcs.cdelt[0],ow.wcs.cdelt[1]]
        w.wcs.ctype     = [ow.wcs.ctype[0],ow.wcs.ctype[1]]
        #
        header          = w.to_header()
        # ===

        # erase 2 axis from original image 
        del_stokes_axis = np.squeeze(sky_image,axis=0)  # delete stokes axis from fits file 
        plane_image     = np.squeeze(del_stokes_axis,axis=0)  # delete frequency axis from fits file
        #
        # === NOTE THE COORDINATES of the array is: plane_image[dec][ra] === IMPORTANT


        # determine FoV verticees
        FoV_verticees = determine_FoV_verticees(plane_image)

        # determine min_flux_density on FoV verticees
        FoV_border_min,FoV_border_max = check_stats_on_verticees(plane_image,FoV_verticees)

        
        # loop through the levels and do the individual estimates 
        #
        for level in range(len(levels)):

            # check if there are pixel in the image that are smaller
            # than the level
            check_image_y,check_image_x = np.where(plane_image <= levels[level])
            

            # set some parameter 
            #
            full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]] = {}


            # determine the field contours
            #
            #
            contours  = measure.find_contours(plane_image,levels[level],fully_connected='low',positive_orientation='low')
            #
            # need to clean up contours that have only 2 verticees
            #
            fcontours             = []
            fcontours_info_add_on = []
            fcontours_info_open   = []
            for k in range(len(contours)):
                #
                # exclude contours that are just lines
                if len(contours[k]) > 2:

                    # get some info
                    percentage_of_FoV_polygon = len(contours[k])/len(FoV_verticees)
                    
                    # get the stats
                    C_border_min,C_border_max = check_stats_on_verticees(plane_image,contours[k])

                    # check if the contours are close
                    cclose, cbounds = check_contour_close(contours[k])

                    if cclose:
                        fcontours.append(contours[k].tolist())
                        #
                        fcontours_info_add_on.append([percentage_of_FoV_polygon,0,0,C_border_min,C_border_max,])                        
                        fcontours_info_open.append(cclose)

                    else:
                        closed_polygon, percentage_of_closeing_polygon, percentage_of_closeing_polygon_FoV  = close_polygon_with_fov(contours[k],FoV_verticees,cbounds)

                        pc_cclose, pc_cbounds = check_contour_close(np.array(closed_polygon))

                        cont_ra,cont_dec = np.array(closed_polygon).T

                        #fig, ax = plt.subplots()
                        #ax.plot(cont_ra,cont_dec,color='orange')
                        #cont_ra , cont_dec  = FoV_verticees.T

                        #ax.plot(cont_ra,cont_dec,color='blue')
                        #plt.show()
                        #sys.exit(-1)

                        # in case something is goin wrong
                        if pc_cclose == False:
                            print('something went wrong closing the contour with the fov verticees')
                            print(' at level ',levels[level])
                            print(k,' cclose ',cclose,'is not close_polygon ',pc_cclose,' ',pc_cbounds)
                            #close_polygon_with_fov(contours[k],FoV_verticees,cbounds)

                            cont_ra,cont_dec = np.array(closed_polygon).T

                            fig, ax = plt.subplots()
                            # check the fov
                            fov_ra , fov_dec  = FoV_verticees.T
                            ax.plot(fov_ra,fov_dec,color='blue')
                            ax.text(fov_ra[0],fov_dec[0],'FoV Start')
                            ax.text(fov_ra[-1],fov_dec[-1],'FoV End')

                            # check the contour
                            #cont_ra , cont_dec  = contours[k].T
                            #ax.scatter(cont_ra,cont_dec,color='orange')
                            #ax.text(cont_ra[0],cont_dec[0],'Cont Start')
                            #ax.text(cont_ra[-1],cont_dec[-1],'Cont End')


                            # check the contour
                            cont_ra , cont_dec  = np.array(closed_polygon).T
                            ax.plot(cont_ra,cont_dec,color='red')
                            ax.text(cont_ra[0],cont_dec[0],'CLOSE Cont Start')
                            ax.text(cont_ra[-1],cont_dec[-1],'CLOSE Cont End')



                            plt.show()
                            sys.exit(-1)
                        #################

                        #
                        fcontours.append(closed_polygon)                        
                        #
                        fcontours_info_add_on.append([percentage_of_FoV_polygon,percentage_of_closeing_polygon,percentage_of_closeing_polygon_FoV,C_border_min,C_border_max,])                        
                        fcontours_info_open.append(pc_cclose)


            #
            # ============


            # Determine if the FoV contours should be included
            #
            if len(check_image_y) != 0:
                FoV_included        =  1
            else:
                FoV_included        =  0
            #
            total_circumverence =  0
            total_patching      =  0
            for fovc in range(len(fcontours_info_add_on)):

                if print_total_area_info == True:
                    print(fovc,fcontours_info_open[fovc],fcontours_info_add_on[fovc],FoV_border_min,FoV_border_max)

                total_circumverence   += fcontours_info_add_on[fovc][0]
                total_patching        += fcontours_info_add_on[fovc][2]


                if fcontours_info_add_on[fovc][0] < 0.1 and fcontours_info_add_on[fovc][2] > 0.9:
                        FoV_included = -1

                # if the contour has not been closed and the stats_min is smaller that the FoV min
                if fcontours_info_add_on[fovc][2] == 0 and fcontours_info_add_on[fovc][3] < FoV_border_min:
                        FoV_included = -1
            
                # here exclude FoV if a single pached contour is close to the FoV length
                if fcontours_info_add_on[fovc][0] > 0.8 and fcontours_info_add_on[fovc][2] > 0:
                    FoV_included = -1


            # overrule if the closing of contours reach more than 80 
            # of the FoV verticees
            #
            if total_circumverence > 0.8 and total_patching > 0.5:
                FoV_included = 1

            #
            # ============


            # add the FoV to the polygons
            if FoV_included == 1:
                fcontours.append(FoV_verticees)

            #
            # ============

            # set a general flag for the polygons
            #
            polygon_check = np.ones(len(fcontours)) 

            #
            # ============


            # 
            # Check contours if they are close 
            # 
            #
            to_check_cont = []
            for i in range(len(fcontours)):
                    cclose, cbounds = check_contour_close(np.array(fcontours[i]))
                    if cclose == False:
                        to_check_cont.append(i)    
            #
            # ============



            # 
            # Check for area coverages
            # and exclude contours within contours
            #

            for i in range(len(fcontours)):
                for j in range(i+1,len(fcontours)):
                        s1 = Polygon(fcontours[i])
                        s2 = Polygon(fcontours[j])

                        if s1.is_valid == False:
                                s1 = s1.buffer(0)
                        if s2.is_valid == False:
                                s2 = s2.buffer(0)
                        
                        intersect = s1.intersection(s2).is_empty

                        if intersect == False:
                            inters_area = s1.intersection(s2).area
                            if inters_area > percentage_area * s1.area:
                                polygon_check[i] = -1
                            if inters_area > percentage_area * s2.area:
                                polygon_check[j] = -1
            #
            # ============


            # another check if contours are within negative contours
            # if so it will not be counted in the area calculations
            #
            for i in range(len(fcontours)):
                for j in range(i+1,len(fcontours)):
                        s1 = Polygon(fcontours[i])
                        s2 = Polygon(fcontours[j])

                        if s1.is_valid == False:
                            s1 = s1.buffer(0)

                        if s2.is_valid == False:
                            s2 = s2.buffer(0)

                        intersect = s1.intersection(s2).is_empty

                        if intersect == False:
                            inters_area = s1.intersection(s2).area
                            if inters_area > percentage_area * s1.area:
                                if polygon_check[i] == -1 and polygon_check[j] == -1:
                                    polygon_check[i] = 0

                            if inters_area > percentage_area * s2.area:
                                if polygon_check[i] == -1 and polygon_check[j] == -1:
                                    polygon_check[j] = 0


            #
            # ============



            #
            # the final polygon check
            # the input can be used to alternate
            # the final area calculations
            #

            doplot_to_check = False
            if len(to_check_cont) > 0:
                doplot_to_check = True

                fkeys = full_info_contours_to_check[fits_file_names[fn].replace('.fits','').replace('.FITS','')].keys()

                if not levels[level] in fkeys:
                    full_info_contours_to_check[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]] = {}

                # print('\nCaution might want to check contours ',np.unique(to_check_cont),'in level ',levels[level],'\n')
                contours_to_check.append([fits_file_names[fn],levels[level],np.ravel(np.unique(to_check_cont)).tolist(),np.ravel(polygon_check).tolist()])

                for cc in to_check_cont:
                    full_info_contours_to_check[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]]['c_'+str(cc)] = -5

            #
            # ============




            #
            # Calculate the area coverage
            #
            if print_total_area_info == True:
                    print('\n== Internal area calculation for level ',levels[level],'\n')
            #
            #
            FoV_area_world = 0 
            for i in range(len(fcontours)):


                full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]]['c_'+str(i)]       = {}

                # change order of coordinates
                # in fcontours because each point is in [dec,ra]
                # keep in nmind: the image coordinates are image[dec][ra]
                #
                im_y,im_x = np.array(fcontours[i]).T  # each point in fcontours is in [dec,ra]
                sky_cont  = np.array([im_x,im_y]).T   # each point is in [ra,dec]

                # Computes the image ra dec into ra dec on the sphere 
                #
                origin = 0   # this is a starnge setting, for numpy array it should be 0 
                Cont_vert_radec_world   = w.wcs_pix2world(sky_cont,origin,ra_dec_order=True)


                # investigate if structure cross 360 degrees in world coordinates 
                # if so split the contours into sub structures (use ellipse_RA_check)
                #
                cont_ra , cont_dec  = Cont_vert_radec_world.T

                #fig, ax = plt.subplots()
                #ax.scatter(cont_ra,cont_dec)
                #plt.show()

                #
                if max(list(np.diff(cont_ra))) > 350:
                    Cont_vert_radec_world = ellipse_RA_check(Cont_vert_radec_world)
                #
                else:
                    Cont_vert_radec_world = [Cont_vert_radec_world]


                # the loop is because ellipse_RA_check can incrase the number of
                # contours, because they will be splitt
                for ac, a in enumerate(Cont_vert_radec_world):

                    full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]]['c_'+str(i)][ac] = {}

                    # The area calculation is based: 
                    #
                    # 1) project the image pixels into world coordinates on a sphere
                    # 2) calculate the angle coverage on that sphere, for this we use
                    #
                    #    polygon_area_on_sphere is based on the earth with 
                    #    input polygon_area_on_sphere(lat,long) so input is dec, ra
                    #
                    # Note just using shapely to calculate the area is wrong
                    # P_area = Polygon(a).area
                    #

                    p_ra,p_dec = np.array(a).T
                    P_area = polygon_area_on_sphere(p_dec,p_ra, algorithm = 0, radius = None)

                    if print_total_area_info == True:
                        print('-- contour ',i,' area ',polygon_check[i],' ',P_area)

                    FoV_area_world += polygon_check[i] * P_area


                    full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]]['c_'+str(i)][ac]['polygon_check'] = polygon_check[i]
                    full_info[fits_file_names[fn].replace('.fits','').replace('.FITS','')][levels[level]]['c_'+str(i)][ac]['polygon_area']  = P_area

            if print_total_area_info == True:
                print('\n- Level ',levels[level],' total area coverage world coordinates ',FoV_area_world)

            tot_area[level]  += FoV_area_world
            tot_area_levels[level] = levels[level]
            #
            # ============




            #
            # Do the plotting in pixel coordinates
            #

            if doplot_pixel_image or doplot_to_check:

                # Some settings for the latex interface
                # https://github.com/matplotlib/matplotlib/issues/5860/
                # https://stackoverflow.com/questions/72140681/matplotlib-set-up-font-computer-modern-and-bold
                #
                plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
                #plt.rc('text', usetex=True)
                #plt.rcParams.update({'font.size': 14})
                #plt.rcParams.update()
                matplotlib.rc('text', usetex=True)
                matplotlib.rc('legend', fontsize=14)
                matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


                # Display the image and plot all contours found
                fig, ax = plt.subplots()

                for i in range(len(fcontours)):

                    if polygon_check[i] > 0:
                        ax.plot(np.array(fcontours[i])[:, 1], np.array(fcontours[i])[:, 0], linewidth=2,color='green')


                    if polygon_check[i] < 0:
                        ax.plot(np.array(fcontours[i])[:, 1], np.array(fcontours[i])[:, 0], linewidth=2,color='red')

                    if polygon_check[i] == 0:    
                        ax.plot(np.array(fcontours[i])[:, 1], np.array(fcontours[i])[:, 0], linewidth=2,color='blue')

                    if do_plot_cont_labels:
                        # do the labeling 
                        ax.text(np.array(fcontours[i])[0, 1], np.array(fcontours[i])[0, 0],str(i),fontsize=10,color='orange')


                    if len(to_check_cont) > 0:
                        if len(np.ravel(np.where(np.unique(to_check_cont) == i))) > 0:
                            ax.plot(np.array(fcontours[i])[:, 1], np.array(fcontours[i])[:, 0], linewidth=3,color='cyan')
                            ax.text(np.array(fcontours[i])[0, 1], np.array(fcontours[i])[0, 0],str(i),color='cyan')


                ax.imshow(plane_image, cmap=plt.cm.gray,norm=matplotlib.colors.LogNorm())


                plt.title(str(fits_file_names[fn].replace('.FITS','').replace('.fits','')+' PIX, level '+str(levels[level])))
                ax.axis('image')
                ax.set_xticks([])
                ax.set_yticks([])

                if pltsave_pixel_image:
                    plt_fname = fits_file_names[fn].replace('.FITS','').replace('.fits','')+'_PIX_Level_'+str(level).zfill(3).replace('.','_')
                    plt_fname = filenamecounter(plt_fname,extention='.png')
                    fig.savefig(plt_fname,dpi=DPI)
                else:
                    plt.show()
                plt.clf()
                plt.cla()
                plt.close()


            #
            # ============





            #
            # Do the plotting in world coordinates
            #
            if doplot_world_contours:
                #
                # only to check is the area is correct
                #

                # Display the image and plot all contours found
                fig, ax = plt.subplots()

                for i in range(len(fcontours)):
                    FoV_vert_radec_world   = w.wcs_pix2world(fcontours[i],0)

                    fov_ra , fov_dec  = FoV_vert_radec_world.T

                    if max(list(np.diff(fov_ra))) > 350:
                        FoV_vert_radec_world = ellipse_RA_check(FoV_vert_radec_world)
                    #
                    else:
                        FoV_vert_radec_world = [FoV_vert_radec_world]

                    for j in range(len(FoV_vert_radec_world)):
                            if polygon_check[i] > 0:
                                ax.text(np.array(FoV_vert_radec_world[j])[0, 0], np.array(FoV_vert_radec_world[j])[0, 1],str(Polygon(FoV_vert_radec_world[j]).area),color='green')
                                ax.scatter(np.array(FoV_vert_radec_world[j])[:, 0], np.array(FoV_vert_radec_world[j])[:, 1], linewidth=2,color='green')
                            if polygon_check[i] < 0:
                                ax.text(np.array(FoV_vert_radec_world[j])[0, 0], np.array(FoV_vert_radec_world[j])[0, 1],str(Polygon(FoV_vert_radec_world[j]).area),color='red')
                                ax.scatter(np.array(FoV_vert_radec_world[j])[:, 0], np.array(FoV_vert_radec_world[j])[:, 1], linewidth=2,color='red')
                            if polygon_check[i] == 0:    
                                ax.text(np.array(FoV_vert_radec_world[j])[0, 0], np.array(FoV_vert_radec_world[j])[0, 1],str(Polygon(FoV_vert_radec_world[j]).area),color='blue')
                                ax.scatter(np.array(FoV_vert_radec_world[j])[:, 0], np.array(FoV_vert_radec_world[j])[:, 1], linewidth=2,color='blue')

                            if len(to_check_cont) > 0:
                                if len(np.ravel(np.where(np.unique(to_check_cont) == i))) > 0:
                                    ax.text(np.array(FoV_vert_radec_world[j])[0, 0], np.array(FoV_vert_radec_world[j])[0, 1],str(Polygon(FoV_vert_radec_world[j]).area),color='cyan')
                                    ax.scatter(np.array(FoV_vert_radec_world[j])[:, 0], np.array(FoV_vert_radec_world[j])[:, 1], linewidth=2,color='cyan')



                plt.title(str(fits_file_names[fn].replace('.FITS','').replace('.fits','')+' WC, level '+str(levels[level])))

                
                if pltsave_world_contours:
                    plt_fname = fits_file_names[fn].replace('.FITS','').replace('.fits','')+'_WC_Level_'+str(level).zfill(3).replace('.','_')
                    plt_fname = filenamecounter(plt_fname,extention='.png')
                    fig.savefig(plt_fname,dpi=DPI)
                else:
                    plt.show()
                plt.clf()
                plt.cla()
                plt.close()

            #
            # ============


    return tot_area, tot_area_levels, contours_to_check, full_info, full_info_contours_to_check




def calculate_full_area(fits_file_names,levels,doplot_pixel_image=False,pltsave_pixel_image=False):
    """
    """

    # use the following settings if you run into a problem

    #
    # Some settings for plotting
    #
    #doplot_pixel_image       = True
    #pltsave_pixel_image      = True
    doplot_world_contours    = False
    pltsave_world_contours   = False

    #
    # These are the default settings
    #
    print_total_area_info    = False
    percentage_area          = 0.98
    do_plot_cont_labels      = True
    DPI                      = 300

    #
    # Here get the machinary started 
    #
    tot_area, tot_area_levels,contours_to_check,full_info,full_info_contours_to_check  = area_coverage(fits_file_names,levels,doplot_pixel_image,\
                                                                                                       pltsave_pixel_image,doplot_world_contours,pltsave_world_contours,\
                                                                                                       print_total_area_info,percentage_area,do_plot_cont_labels,DPI)


    no_bad_contours = -1
    docheck_for_bad_contours = True
    if docheck_for_bad_contours:
        if print_total_area_info:
            print('\n\n=== Possible bad contours to be checked ===\n')
        data_keys = full_info_contours_to_check.keys()


        for ky in data_keys:
            level_keys = full_info_contours_to_check[ky].keys()

            if len(level_keys) == 0:
                #    print('- No bad contours found.\n')
                no_bad_contours = True

            for l in level_keys:
                    contour_keys = full_info_contours_to_check[ky][l].keys()
                    for c in contour_keys:
                            sub_contour_keys = full_info[ky][l][c].keys()
                            for sc in sub_contour_keys:
                                    print(ky,l,c,full_info_contours_to_check[ky][l][c],full_info[ky][l][c][sc]['polygon_check'],full_info[ky][l][c][sc]['polygon_area'])



    docalculate_area = True
    if docalculate_area:

        if print_total_area_info:
            print('\n============ Calculate area ==============\n')

        do_prt_info = False

        # here some settings if you need to correct something by hand
        #
        do_polygon_check_by_hand = 0
        #
        # just edit the full_info_contours_to_check
        #
        # full_info_contours_to_check['J1312-2026_rms'][0.0001291549665014884]['c_0'] = 1
        # full_info_contours_to_check['J2023-3655_rms'][0.0001668100537200059]['c_0'] = -1
        # full_info_contours_to_check['J006+1728_rms'][0.0001668100537200059]['c_0'] = 1

        #
        # Calculate the full area coverage per level
        #
        full_area_info    = OrderedDict()
        #
        #
        data_keys = full_info.keys()
        for ky in data_keys:
            level_keys = full_info[ky].keys()

            for l in level_keys:
                    if not l in full_area_info:
                            full_area_info[l] = {}
                            full_area_info[l]['area'] = 0
                            full_area_info[l]['area_error'] = 0
                    #
                    #
                    sky_area = 0
                    sky_area_error = 0
                    contour_keys = full_info[ky][l].keys()
                    for c in contour_keys:
                        sub_contour_keys = full_info[ky][l][c].keys()

                        for sc in sub_contour_keys:
                            area          = full_info[ky][l][c][sc]['polygon_area']
                            polygon_check = full_info[ky][l][c][sc]['polygon_check']

                            if ky in full_info_contours_to_check:
                                if l in full_info_contours_to_check[ky]:
                                    if c in full_info_contours_to_check[ky][l]:
                                        if full_info_contours_to_check[ky][l][c] != -5:
                                            polygon_check = full_info_contours_to_check[ky][l][c]
                                        else:
                                            if polygon_check != 0:
                                                print('exclude ',ky,l,c,sc)
                                                polygon_check = do_polygon_check_by_hand
                                                sky_area_error += area
                            if do_prt_info:
                                print(ky,l,c,sc,polygon_check,area)

                            sky_area += polygon_check * area

                    full_area_info[l]['area'] += sky_area
                    full_area_info[l]['area_error'] += sky_area_error



    return full_area_info, no_bad_contours


def make_example():

    #
    # Estimate the sky coverage 
    #
    doplot_pixel_image  = False
    pltsave_pixel_image = False
    #
    levels = np.logspace(-5,1,55)[10:12]
    #
    fits_file_names = ['J0006+1728_rms.fits']

    # ==========================

    # get the calculation going
    full_area_info, no_bad_contours = calculate_full_area(fits_file_names,levels,doplot_pixel_image,pltsave_pixel_image)



def calculate_survey_area(fits_file_names,levels,doplot_pixel_image=False,pltsave_pixel_image=False,doplot_area_p_level=False):

    #
    # Estimate the sky coverage 
    #
    # doplot_pixel_image  = False
    # pltsave_pixel_image = False
    # doplot_area_p_level = False
    #
    # levels = np.logspace(-5,1,55)
    #
    # fits_file_names = ['J0001-1540_rms.fits','J0006+1728_rms.fits','J0126+1420_rms.fits','J0240+0957_rms.fits','J0249+0440_rms.fits',\
    #                      'J0249-0759_rms.fits','J1133+0015_rms.fits','J1232-0224_rms.fits','J1312-2026_rms.fits','J2023-3655_rms.fits']

    # ==========================


    # get the calculation going
    full_area_info, no_bad_contours = calculate_full_area(fits_file_names,levels,doplot_pixel_image,pltsave_pixel_image)

    #
    if no_bad_contours == False:
        print('Need to check the levels someting went wrong')


    tot_area_levels = []
    tot_area        = []
    tot_area_error  = []

    for l in full_area_info:
        tot_area_levels.append(l)
        tot_area.append(full_area_info[l]['area'])
        tot_area_error.append(full_area_info[l]['area_error'])

    # do the plotting
    #
    if doplot_area_p_level:

        fig, ax = plt.subplots()
        ax.errorbar(tot_area_levels,tot_area,yerr=tot_area_error,fmt='o')
        ax.set_xscale('log',base=10)
        plt.show()

    return tot_area_levels, tot_area, tot_area_error



# comment out if you want to run the example
#
# make_example()
#
