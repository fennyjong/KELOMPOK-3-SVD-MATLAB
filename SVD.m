clc; clear; close all;

%%% Training process %%%
% 1. Load image file
nama_folder = 'data latih';
nama_file = dir(fullfile(nama_folder,'*.jpg'));
jumlah_file = numel(nama_file);
% 2. initializing the training_feature variable
ciri_latih = zeros(jumlah_file,7);

for n = 1:jumlah_file
    % 3. reading an RGB image 
    Img = imread(fullfile(nama_folder,nama_file(n).name));
    % 4. converting an RGB image to grayscale
    Img_gray = rgb2gray(Img);
    % 5. converting a grayscale image to binary
    bw = im2bw(Img_gray,graythresh(Img_gray));
    % 6. morphological operation
    bw = imcomplement(bw);
    bw = imfill(bw,'holes');
    bw = bwareaopen(bw,100);
    % 7. RGB color feature extraction
    R = Img(:,:,1);
    G = Img(:,:,2);
    B = Img(:,:,3);
    R(~bw) = 0;
    G(~bw) = 0;
    B(~bw) = 0;
    Red = sum(sum(R))/sum(sum(bw));
    Green = sum(sum(G))/sum(sum(bw));
    Blue = sum(sum(B))/sum(sum(bw));
    % 8. HSV color feature extraction
    HSV = rgb2hsv(Img);
    H = HSV(:,:,1);
    S = HSV(:,:,2);
    V = HSV(:,:,3);
    H(~bw) = 0;
    S(~bw) = 0;
    V(~bw) = 0;
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    % 9. size feature extraction
    Area = sum(sum(bw));
    % 10. filling the extracted feature results into the training_feature variable
    ciri_latih(n,1) = Red;
    ciri_latih(n,2) = Green;
    ciri_latih(n,3) = Blue;
    ciri_latih(n,4) = Hue;
    ciri_latih(n,5) = Saturation;
    ciri_latih(n,6) = Value;
    ciri_latih(n,7) = Area;
end

% 11. normalization data
[ciri_latihZ,muZ,sigmaZ] = zscore(ciri_latih);

% 12. dimensionality reduction with SVD
[U, S, V] = svd(ciri_latihZ);
k = 7; % jumlah komponen yang dipilih (misalnya 7)
ciri_latihSVD = U(:,1:k) * S(1:k,1:k);

% 13. initializing the training_class variable
kelas_latih = cell(jumlah_file,1);
% 14. filling vegetable names into the training_class variable
for k = 1:12
    kelas_latih{k} = 'brokoli';
end

for k = 13:27
    kelas_latih{k} = 'cabe';
end

for k = 28:47
    kelas_latih{k} = 'kentang';
end

for k = 48:54
    kelas_latih{k} = 'kol';
end

for k = 55:58
    kelas_latih{k} = 'sawi';
end

for k = 59:68
    kelas_latih{k} = 'tomat';
end

for k = 69:73
    kelas_latih{k} = 'wortel';
end


% 15. extract SVD1 & SVD2
SVD1 = ciri_latihSVD(:,1);
SVD2 = ciri_latihSVD(:,2);

% Broccoli class 
x1 = SVD1(1:12);
y1 = SVD2(1:12);

% Chili class 
x2 = SVD1(13:27);
y2 = SVD2(13:27);

% Potato class 
x3 = SVD1(28:47);
y3 = SVD2(28:47);

% Cabbage class 
x4 = SVD1(48:54);
y4 = SVD2(48:54);

% Chinese Cabbage class 
x5 = SVD1(55:58);
y5 = SVD2(55:58);

% Tomato class 
x6 = SVD1(59:68);
y6 = SVD2(59:68);

% Carrot class 
x7 = SVD1(69:73);
y7 = SVD2(69:73);

% 16. displaying data distribution in each training class
figure
plot(x1,y1,'r.','MarkerSize',30)
hold on
plot(x2,y2,'g.','MarkerSize',30)
plot(x3,y3,'b.','MarkerSize',30)
plot(x4,y4,'m.','MarkerSize',30)
plot(x5,y5,'c.','MarkerSize',30)
plot(x6,y6,'y.','MarkerSize',30)
plot(x7,y7,'k.','MarkerSize',30)
hold off
grid on
xlabel('SVD1')
ylabel('SVD2')
legend('Broccoli', 'Chili', 'Potato', 'Cabbage', 'Chinese Cabbage', 'Tomato', 'Carrot')
title('Data Distribution in Training Classes with SVD')


% 17. classification using k-Nearest Neighbors (k-NN)
Mdl = fitcknn([SVD1,SVD2],kelas_latih,'NumNeighbors',7);

% Print the results
disp('Training Data:')
disp('Image   Red   Green   Blue   Hue   Saturation   Value   Area')
disp('--------------------------------------------------------------')
for n = 1:jumlah_file
    disp([num2str(n) '      ' num2str(ciri_latih(n,1), '%.2f') '   ' num2str(ciri_latih(n,2), '%.2f') '   ' num2str(ciri_latih(n,3), '%.2f') '   ' num2str(ciri_latih(n,4), '%.2f') '   ' num2str(ciri_latih(n,5), '%.2f') '   ' num2str(ciri_latih(n,6), '%.2f') '   ' num2str(ciri_latih(n,7))])
end

%%% Testing Proccess %%%
% 1. Load image file
nama_folder = 'data uji';
nama_file = dir(fullfile(nama_folder,'*.jpg'));
jumlah_file = numel(nama_file);
% 2. initializing the test_feature variable
ciri_uji = zeros(jumlah_file,7);

for n = 1:jumlah_file
    % 3. reading an RGB image 
    Img = imread(fullfile(nama_folder,nama_file(n).name));
    % 4. converting an RGB image to grayscale
    Img_gray = rgb2gray(Img);
    % 5. converting a grayscale image to binary
    bw = im2bw(Img_gray,graythresh(Img_gray));
    % 6. morphological operation
    bw = imcomplement(bw);
    bw = imfill(bw,'holes');
    bw = bwareaopen(bw,100);
    % 7. RGB color feature extraction
    R = Img(:,:,1);
    G = Img(:,:,2);
    B = Img(:,:,3);
    R(~bw) = 0;
    G(~bw) = 0;
    B(~bw) = 0;
    Red = sum(sum(R))/sum(sum(bw));
    Green = sum(sum(G))/sum(sum(bw));
    Blue = sum(sum(B))/sum(sum(bw));
    % 8. HSV color feature extraction
    HSV = rgb2hsv(Img);
    H = HSV(:,:,1);
    S = HSV(:,:,2);
    V = HSV(:,:,3);
    H(~bw) = 0;
    S(~bw) = 0;
    V(~bw) = 0;
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    % 9. size feature extraction
    Area = sum(sum(bw));
    % 10. filling the extracted feature results into the test_feature variable
    ciri_uji(n,1) = Red;
    ciri_uji(n,2) = Green;
    ciri_uji(n,3) = Blue;
    ciri_uji(n,4) = Hue;
    ciri_uji(n,5) = Saturation;
    ciri_uji(n,6) = Value;
    ciri_uji(n,7) = Area;
end

% 11. Standardizing the test features
ciri_ujiZ = zeros(jumlah_file,7);
for k = 1:jumlah_file
    ciri_ujiZ(k,:) = (ciri_uji(k,:) - muZ)./sigmaZ;
end

% 12. dimensionality reduction with SVD
[U, S, V] = svd(ciri_ujiZ);

% 13. extract SVD1 & SVD2
SVD1 = ciri_ujiZ(:,1);
SVD2 = ciri_ujiZ(:,2);

% 14. testing the test data on k-Nearest Neighbors (k-NN)
hasil_uji = predict(Mdl,[SVD1,SVD2]);

% 15. displaying the data distribution in each training class
figure
plot(x1,y1,'r.','MarkerSize',30)
hold on
plot(x2,y2,'g.','MarkerSize',30)
plot(x3,y3,'b.','MarkerSize',30)
plot(x4,y4,'m.','MarkerSize',30)
plot(x5,y5,'c.','MarkerSize',30)
plot(x6,y6,'y.','MarkerSize',30)
plot(x7,y7,'k.','MarkerSize',30)
grid on

% 16. displaying the data distribution in each test class

% Category: brokoli
plot(SVD1(string(hasil_uji)=='brokoli'), SVD2(string(hasil_uji)=='brokoli'), 'rx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: cabe
plot(SVD1(string(hasil_uji)=='cabe'), SVD2(string(hasil_uji)=='cabe'), 'gx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: kentang
plot(SVD1(string(hasil_uji)=='kentang'), SVD2(string(hasil_uji)=='kentang'), 'bx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: kol
plot(SVD1(string(hasil_uji)=='kol'), SVD2(string(hasil_uji)=='kol'), 'mx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: sawi
plot(SVD1(string(hasil_uji)=='sawi'), SVD2(string(hasil_uji)=='sawi'), 'cx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: tomat
plot(SVD1(string(hasil_uji)=='tomat'), SVD2(string(hasil_uji)=='tomat'), 'yx', 'LineWidth', 2, 'MarkerSize', 10);

% Category: wortel
plot(SVD1(string(hasil_uji)=='wortel'), SVD2(string(hasil_uji)=='wortel'), 'kx', 'LineWidth', 2, 'MarkerSize', 10);

hold off

xlabel('SVD1')
ylabel('SVD2')

legend('Broccoli (latih)', 'Chili (latih)', 'Potato (latih)', 'Cabbage (latih)', 'Chinese Cabbage (latih)', 'Tomato (latih)', 'Carrot (latih)',...
    'Broccoli (uji)', 'Chili (uji)', 'Potato (uji)','Cabbage (uji)', 'Tomato (uji)', 'Carrot (uji)')
title('Data Distribution in Training and Test Classes with SVD')


fprintf('\n');
disp('Testing Data:')
disp('Image   Red   Green   Blue   Hue   Saturation   Value   Area')
disp('--------------------------------------------------------------')
for n = 1:jumlah_file
    disp([num2str(n) '      ' num2str(ciri_uji(n,1), '%.2f') '   ' num2str(ciri_uji(n,2), '%.2f') '   ' num2str(ciri_uji(n,3), '%.2f') '   ' num2str(ciri_uji(n,4), '%.2f') '   ' num2str(ciri_uji(n,5), '%.2f') '   ' num2str(ciri_uji(n,6), '%.2f') '   ' num2str(ciri_uji(n,7))])
end

% 17. Perform image segmentation on the test images
segmented_images = cell(jumlah_file, 1);

for n = 1:jumlah_file
    % Read the test image
    Img = imread(fullfile(nama_folder, nama_file(n).name));
    
    % Convert the test image to grayscale
    Img_gray = rgb2gray(Img);
    
    % Convert the grayscale image to binary
    bw = im2bw(Img_gray, graythresh(Img_gray));
    
    % Perform morphological operations
    bw = imcomplement(bw);
    bw = imfill(bw, 'holes');
    bw = bwareaopen(bw, 100);
    
    % Apply segmentation to the original RGB image
    segmented_image = Img;
    segmented_image(repmat(~bw, [1, 1, 3])) = 0;
    
    % Store the segmented image in the cell array
    segmented_images{n} = segmented_image;
end

% Display the segmented images
for n = 1:jumlah_file
    figure;
    imshow(segmented_images{n});
    title(sprintf('Segmented Image %d', n));
    set(gcf, 'Position', get(0, 'Screensize')); % Memperbesar gambar ke ukuran layar
end