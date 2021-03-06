function [ X, D, L ] = loadDataSet( dataSetNr )
% LOADDATASET Loads data samples (X) , training targets (D) and labels (L)
% Samples are in the 1st dimension (rows), and features in the
% 2nd dimension. This convention must be consistent throuhout the
% assignment, otherwise the plot code will break.

switch dataSetNr
    case 1
        load lab_data
        X = [X1'; Xt1'];
        L = [L1; Lt1];
        D = [D1'; Dt1'];
	case 2
        load lab_data
        X = [X2'; Xt2'];
        L = [L2; Lt2];
        D = [D2'; Dt2'];
    case 3
        load lab_data
        X = [X3'; Xt3'];
        L = [L3; Lt3];
        D = [D3'; Dt3'];
    case 4
        load lab_data_digits
        L = [L; Lt] + 1;
        X = [X'; Xt'];
        D = [D'; Dt'];
        
        
    otherwise
        error('Unknown DataSet Number')
end

end

