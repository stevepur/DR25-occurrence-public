import numpy as np

def evaluateModel(x, y, theta, xRange, yRange, model):
    xx, yy = normalizeRange(np.double(x), np.double(y), xRange, yRange);
    return rateModel(xx, yy, theta, model);

def normalizeRange(x, y, xRange, yRange):
    return (x - xRange[0])/(xRange[1] - xRange[0]), (y - yRange[0])/(yRange[1] - yRange[0])

def rateModel(x, y, theta, model):
    if model == "constant":
        r = theta*np.ones(np.shape(x));
    elif model == "linearX":
        m,b = theta;
        r = m*x + b;
    elif model == "linearXY":
        mx, my, b = theta;
        r = mx*x + my*y + b;
    elif model == "gaussian":
        x0, y0, sx, sy, amp, b = theta;
        r = amp*np.exp(-((x-x0)**2/(2*sx*sx) + (y-y0)**2/(2*sy*sy))) + b;
    elif model == "logisticX":
        x0, k, amp, b = theta;
        r = amp/(1 + np.exp(-k*(x-x0))) + b;
    elif model == "logisticY":
        y0, k, amp, b = theta;
        r = amp/(1 + np.exp(-k*(y-y0))) + b;
    elif model == "rotatedLogisticY":
        y0, k, phi, amp, b = theta;
        phiRad = np.pi*phi/180;
        yr = (y-0.5)*np.cos(phiRad) - (x-0.5)*np.sin(phiRad);
        r = amp/(1 + np.exp(-k*(yr+0.5-y0))) + b;
    elif model == "rotatedLogisticY2":
        y0, k, nu, phi, amp, b = theta;
        phiRad = np.pi*phi/180;
        yr = (y-0.5)*np.cos(phiRad) - (x-0.5)*np.sin(phiRad);
        r = amp/((1 + np.exp(-k*(yr+0.5-y0)))**(1/nu)) + b;
    elif model == "rotatedLogisticYXFixedLogisticY":
        y0, k, phi, amp, b = theta;
        phiRad = np.pi*phi/180;
        yr = (y-0.5)*np.cos(phiRad) - (x-0.5)*np.sin(phiRad);
        r = amp/((1 + np.exp(-k*(yr+0.5-y0)))*(1 + np.exp(-33.331*(y-(-0.25))))) + b;
    elif model == "rotatedLogisticYXLogisticY":
        y0, k, y02, k2, phi, amp, b = theta;
        phiRad = np.pi*phi/180;
        yr = (y-0.5)*np.cos(phiRad) - (x-0.5)*np.sin(phiRad);
        r = amp/((1 + np.exp(-k*(yr+0.5-y0)))*(1 + np.exp(-k2*(y-y02)))) + b;
    elif model == "logisticY2":
        y0, k, nu, amp, b = theta;
        r = amp/((1 + np.exp(-k*(y-y0)))**(1/nu)) + b;
    elif model == "logisticX0":
        x0, k, amp = theta;
        r = amp/(1 + np.exp(k*(x-x0)));
    elif model == "logisticY0":
        y0, k, amp = theta;
        r = amp/(1 + np.exp(-k*(y-y0)));
    elif model == "logisticY02":
        y0, k, nu, amp = theta;
        r = amp/((1 + np.exp(-k*(y-y0)))**(1/nu));
    elif model == "logisticX0xlogisticY0":
        x0, y0, kx, ky, amp = theta;
        r = amp/((1 + np.exp(kx*(x-x0)))*(1 + np.exp(-ky*(y-y0))));
    elif model == "logisticX0xlogisticY02":
        x0, y0, kx, ky, nux, nuy, amp = theta;
        r = amp/(((1 + np.exp(kx*(x-x0)))**(1/nux))*((1 + np.exp(-ky*(y-y0)))**(1/nuy)));
    elif model == "logisticX0xRotatedLogisticY0":
        x0, y0, kx, ky, phiy, amp = theta;
        phiRady = np.pi*phiy/180;
        yr = (y-0.5)*np.cos(phiRady) - (x-0.5)*np.sin(phiRady);
        r = amp/((1 + np.exp(kx*(x-x0)))*(1 + np.exp(-ky*(yr+0.5-y0))));
    elif model == "logisticX0xRotatedLogisticY02":
        x0, y0, kx, ky, nuy, phiy, amp = theta;
        phiRady = np.pi*phiy/180;
        yr = (y-0.5)*np.cos(phiRady) - (x-0.5)*np.sin(phiRady);
        r = amp/((1 + np.exp(kx*(x-x0)))*((1 + np.exp(-ky*(yr+0.5-y0)))**(1/nuy)));
    elif model == "rotatedLogisticX0xlogisticY0":
        x0, y0, kx, ky, phix, phiy, amp = theta;
        phiRadx = np.pi*phix/180;
        xr = (x-0.5)*np.cos(phiRadx) - (y-0.5)*np.sin(phiRadx);
        phiRady = np.pi*phiy/180;
        yr = (y-0.5)*np.cos(phiRady) - (x-0.5)*np.sin(phiRady);
        r = amp/((1 + np.exp(kx*(xr+0.5-x0)))*(1 + np.exp(-ky*(yr+0.5-y0))));
    elif model == "rotatedLogisticX0xlogisticY02":
        x0, y0, kx, ky, nux, nuy, phix, phiy, amp = theta;
        phiRadx = np.pi*phix/180;
        xr = (x-0.5)*np.cos(phiRadx) - (y-0.5)*np.sin(phiRadx);
        phiRady = np.pi*phiy/180;
        yr = (y-0.5)*np.cos(phiRady) - (x-0.5)*np.sin(phiRady);
        r = amp/(((1 + np.exp(kx*(xr+0.5-x0)))**(1/nux))*((1 + np.exp(-ky*(yr+0.5-y0)))**(1/nuy)));
    elif model == "rotatedLogisticX0":
        x0, k, amp, phi = theta;
        phiRad = np.pi*phi/180;
        xr = (x-0.5)*np.cos(phiRad) - (y-0.5)*np.sin(phiRad);
        r = amp/(1 + np.exp(k*(xr + 0.5 - x0)));
    elif model == "rotatedLogisticX0+gaussian":
        x0, k, amp, phi, gx0, gy0, gsx, gsy, gamp = theta;
        r = gamp*np.exp(-((x-gx0)**2/(2*gsx*gsx) + (y-gy0)**2/(2*gsy*gsy)));
        phiRad = np.pi*phi/180;
        xr = (x-0.5)*np.cos(phiRad) - (y-0.5)*np.sin(phiRad);
        r = r + amp/(1 + np.exp(k*(xr + 0.5 - x0)));
    elif model == "rotatedLogisticX02":
        x0, k, nu, amp, phi = theta;
        phiRad = np.pi*phi/180;
        xr = (x-0.5)*np.cos(phiRad) - (y-0.5)*np.sin(phiRad);
        r = amp/(1 + np.exp(k*(xr + 0.5 - x0)))**(1/nu);
    elif model == "dualBrokenPowerLaw":
        xb, yb, ax, bx, ay, by, amp = theta;
        # if x is a scalar float we gotta turn it into an array
        if np.isscalar(x) == True:
            xArray = np.asarray([x]);
            yArray = np.asarray([y]);
        else:
            xArray = x;
            yArray = y;

        rx = np.zeros(np.shape(xArray));
        x1Idx = np.where(xArray < xb);
        x2Idx = np.where(xArray >= xb);
        rx[x1Idx] = ((xArray[x1Idx] + 1)/(xb+1))**ax;
        rx[x2Idx] = ((xArray[x2Idx] + 1)/(xb+1))**bx;

        ry = np.zeros(np.shape(xArray));
        x1Idx = np.where(yArray < yb);
        x2Idx = np.where(yArray >= yb);
        ry[x1Idx] = ((yArray[x1Idx] + 1)/(yb+1))**ay;
        ry[x2Idx] = ((yArray[x2Idx] + 1)/(yb+1))**by;

        r = amp*rx*ry;
    else:
        raise ValueError('Bad model name');
    
    if np.isscalar(r) == True:
        if r < 0:
            r = 0;
    else:
        r[r<0] = 0;
    
    return r

def getModelLabels(model):
    if model == "constant":
        return ["r"];
    elif model == "linearX":
        return ["m", "b"];
    elif model == "linearXY":
        return ["mx", "my", "b"];
    elif model == "gaussian":
        return ["x0", "y0", "$\sigma_x$", "$\sigma_y$", "A", "b"];
    elif model == "logisticX":
        return ["x0", "k", "A", "b"];
    elif model == "logisticY":
        return ["y0", "k", "A", "b"];
    elif model == "rotatedLogisticY":
        return ["y0", "k", "phi", "A", "b"];
    elif model == "rotatedLogisticY2":
        return ["y0", "k", "$\\nu$", "$\phi$", "A", "b"];
    elif model == "rotatedLogisticYXFixedLogisticY":
        return ["y0", "k", "phi", "A", "b"];
    elif model == "rotatedLogisticYXLogisticY":
        return ["y0", "k", "y02", "k2", "phi", "A", "b"];
    elif model == "logisticY2":
        return ["y0", "k", "nu", "A", "b"];
    elif model == "logisticX0":
        return ["x0", "k", "A"];
    elif model == "logisticY0":
        return ["y0", "k", "A"];
    elif model == "logisticY02":
        return ["y0", "k", "nu", "A"];
    elif model == "logisticX0xlogisticY0":
        return ["x0", "y0", "kx", "ky", "A"];
    elif model == "logisticX0xlogisticY02":
        return ["x0", "y0", "kx", "ky", "nux", "nuy", "A"];
    elif model == "logisticX0xRotatedLogisticY0":
        return ["$x_0$", "$y_0$", "$k_x$", "$k_y$", "$\phi$", "$A$"];
    elif model == "logisticX0xRotatedLogisticY02":
        return ["$x_0$", "$y_0$", "$k_x$", "$k_y$", "$\\nu$", "$\phi$", "$A$"];
    elif model == "rotatedLogisticX0xlogisticY0":
        return ["x0", "y0", "kx", "ky", "phix", "phiy", "A"];
    elif model == "rotatedLogisticX0xlogisticY02":
        return ["x0", "y0", "kx", "ky", "nux", "nuy", "phix", "phiy", "A"];
    elif model == "rotatedLogisticX0":
        return ["$x_0$", "$k_x$", "$A$", "$\phi$"];
    elif model == "rotatedLogisticX02":
        return ["x0", "k", "nu", "A", "phi"];
    elif model == "rotatedLogisticX0+gaussian":
        return ["x0", "k", "A", "phi", "gx0", "gy0", "$\sigma_x$", "$\sigma_y$", "gAmp"];
    elif model == "dualBrokenPowerLaw":
        return ["xb", "yb", "ax", "bx", "ay", "by", "A"];
    else:
        raise ValueError('Bad model name');

def initRateModel(model):
    if model == "constant":
        theta = [0.9];
    elif model == "linearX":
        minRate = 0.1;
        maxRate = 0.99;
        m = -(maxRate - minRate); # we're on unit square, so delta x = 1
        b = maxRate; # max at x = 0
        theta = [m, b];
    elif model == "linearXY":
        minRate = 0.1;
        maxRate = 0.9;
        # be careful not to let the model go < 0
        mx = -0.5*(maxRate - minRate); # we're on unit square, so delta x = 1
        my = -0.5*(maxRate - minRate); # we're on unit square, so delta x = 1
        b = maxRate; # max at x, y = 0
        theta = [mx, my, b];
    elif model == "gaussian":
        # be careful not to let the model go < 0
        x0 = 0.5;
        y0 = 0.4;
        sx = 0.2;
        sy = 0.2;
        amp = -0.1;
        b = 1.0;
        theta = [x0, y0, sx, sy, amp, b];
    elif model == "logisticX":
        minRate = 0.1;
        maxRate = 1.0;
        # be careful not to let the model go < 0
        x0 = 0.3;
        k = 20;
        amp = maxRate - minRate;
        b = minRate;
        theta = [x0, k, amp, b];
    elif model == "logisticY":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, b, amp];
    elif model == "rotatedLogisticY":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        phi = 0.0;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, phi, b, amp];
    elif model == "rotatedLogisticY2":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        nu = 1;
        phi = 0.0;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, nu, phi, b, amp];
    elif model == "rotatedLogisticYXFixedLogisticY":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        phi = 0.0;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, phi, b, amp];
    elif model == "rotatedLogisticYXLogisticY":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        y02 = 0.5;
        k2 = 2;
        phi = 0.0;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, y02, k2, phi, b, amp];
    elif model == "logisticY2":
        minRate = 0.1;
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        nu = 1;
        amp = maxRate - minRate;
        b = minRate;
        theta = [y0, k, nu, b, amp];
    elif model == "logisticX0":
        maxRate = 1.0;
        x0 = 0.3;
        k = 20;
        amp = maxRate;
        theta = [x0, k, amp];
    elif model == "logisticY0":
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        amp = maxRate;
        theta = [y0, k, amp];
    elif model == "logisticY02":
        maxRate = 1.0;
        y0 = 0.5;
        k = 2;
        nu = 1;
        amp = maxRate;
        theta = [y0, k, nu, amp];
    elif model == "logisticX0xlogisticY0":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        amp = maxRate;
        theta = [x0, y0, kx, ky, amp];
    elif model == "logisticX0xlogisticY02":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        nux = 1.0;
        nuy = 1.0;
        amp = maxRate;
        theta = [x0, y0, kx, ky, nux, nuy, amp];
    elif model == "logisticX0xRotatedLogisticY0":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        phiy = 0;
        amp = maxRate;
        theta = [x0, y0, kx, ky, phiy, amp];
    elif model == "logisticX0xRotatedLogisticY02":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        nuy = 1.0;
        phiy = 0;
        amp = maxRate;
        theta = [x0, y0, kx, ky, nuy, phiy, amp];
    elif model == "rotatedLogisticX0xlogisticY0":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        phix = 0;
        phiy = 0;
        amp = maxRate;
        theta = [x0, y0, kx, ky, phix, phiy, amp];
    elif model == "rotatedLogisticX0xlogisticY02":
        maxRate = 1.0;
        x0 = 0.8;
        kx = 0.3;
        y0 = 0.3;
        ky = 15;
        nux = 1.0;
        nuy = 1.0;
        phix = 0;
        phiy = 0;
        amp = maxRate;
        theta = [x0, y0, kx, ky, nux, nuy, phix, phiy, amp];
    elif model == "rotatedLogisticX0":
        maxRate = 1.0;
        x0 = 0.5;
        k = 8;
        amp = maxRate;
        phi = 107;
        theta = [x0, k, amp, phi];
    elif model == "rotatedLogisticX02":
        maxRate = 1.0;
        x0 = 0.5;
        k = 8;
        nu = 1;
        amp = maxRate;
        phi = 107;
        theta = [x0, k, nu, amp, phi];
    elif model == "rotatedLogisticX0+gaussian":
        gx0 = 0.5;
        gy0 = 0.4;
        gsx = 0.2;
        gsy = 0.2;
        gamp = -0.1;
        maxRate = 1.0;
        x0 = 0.5;
        k = 8;
        amp = maxRate;
        phi = 107;
        theta = [x0, k, amp, phi, gx0, gy0, gsx, gsy, gamp];
    elif model == "dualBrokenPowerLaw":
        xb = 0.6;
        yb = 0.3;
        ax = -0.07;
        bx = -0.4
        ay = 1;
        by = 0.1
        amp = 0.63;
        theta = [xb, yb, ax, bx, ay, by, amp];
    else:
        raise ValueError('Bad model name');
    
    return theta
