function GaussPlot(mu1,sig1,mu2,sig2,i)
    subi = mod(i,16);
    if (subi== 0)
        subi = 16;
    end
    subplot(4,4,subi); 
    xmin = min(mu1,mu2)-2*(sig1+sig2);
    xmax = max(mu1,mu2)+2*(sig1+sig2);
    x = xmin:(xmax-xmin)/100:xmax;
    y1 = (sqrt(2*pi)*sig1).^(-1) * exp(-(x-mu1).^2/(2*sig1*sig1));
    y2 = (sqrt(2*pi)*sig2).^(-1) * exp(-(x-mu2).^2/(2*sig2*sig2));
    plot(x,y1);
    hold on
    plot(x,y2);
    title(i);
end