close all; clearvars; clc;

% First, this script imports the text file data_3D_METHOD.txt, which 
% contains the movement log of the surgical maneuvers. Then, it translates 
% and rotates the 3D points to align with the perspective of the real model. 
% NOTE: The MAPs calculated in the following scripts are invariant to 
% translation and rotation.



if matlab.desktop.editor.isEditorAvailable
    full_path = matlab.desktop.editor.getActiveFilename;
else
    full_path = mfilename('fullpath');
end 
                                                                                  
[path_script, ~, ~] = fileparts(full_path); 

analisis = load(strcat(path_script, '\data_3D_mitracks.txt'));  % Or data_3D_YOLO8.txt    

%% Starting

fs = 30;

tam=size(analisis);
ini = 33;                     
fin = tam(1)-33;                       

verdet = analisis(ini:fin,1:3); % Right hand x,y,z data
azult =  analisis(ini:fin,4:6); % Left hand x,y,z data

verdet = verdet.*10;    % cm to mm
azult = azult.*10;


%% Rotational
% If the original rotation changes, recalibrate with data3D_centro.mat

ax=40; ay=5; az=0;                % Mitracks3D configuration

MRx = [1 0 0; 0 cosd(ax) -sind(ax); 0 sind(ax) cosd(ax)];
MRy = [cosd(ay) 0 sind(ay); 0 1 0; -sind(ay) 0 cosd(ay)];
MRz = [cosd(az) -sind(az) 0; sind(az) cosd(az) 0; 0 0 1]; 

samples = size(azult, 1);
tiempo=0:1/fs:(1/fs)*(samples-1); 

for i = 1: samples

        rot = MRx*(verdet(i,:).');
        rot = MRy*rot;
        verdet(i,:) = rot.';

        rot2 = MRx*(azult(i,:).');
        rot2 = MRy*rot2;
        azult(i,:) = rot2.';

end


verdet(:,3) = verdet(:,3).*-1;
azult(:,3) = azult(:,3).*-1;

% With reference to the physical model, set to 0 in absolute coordinates
verdet(:,1) = verdet(:,1) - 28;    
verdet(:,2) = verdet(:,2) + 86;
verdet(:,3) = verdet(:,3) + 124;
azult(:,1) = azult(:,1) - 28;
azult(:,2) = azult(:,2) + 86;
azult(:,3) = azult(:,3) + 124;
ven = 15;          
verdet(:,1) = movmean(verdet(:,1), ven);
verdet(:,2) = movmean(verdet(:,2), ven);
verdet(:,3) = movmean(verdet(:,3), ven);
azult(:,1) = movmean(azult(:,1), ven);
azult(:,2) = movmean(azult(:,2), ven);
azult(:,3) = movmean(azult(:,3), ven);

MuestraTR(:,1:3) = verdet;
MuestraTR(:,4:6) = azult;
MuestraTR(:,7) = tiempo;


% Save the data 3-3-1
Formated_MuestraTR = sprintfc('%.3f', MuestraTR);
writecell(Formated_MuestraTR, strcat(path_script, '\output_Data3D_2b_analized.txt'), 'Delimiter', 'tab');
% This script should be used on each 3D motion register, it can be modified
% to be a function.

