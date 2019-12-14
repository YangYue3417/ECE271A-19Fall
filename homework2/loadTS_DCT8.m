function [BG, FG] = loadTS_DCT8(dtype,text)
    load(dtype,text);
    BG = TrainsampleDCT_BG;
    FG = TrainsampleDCT_FG;
end
