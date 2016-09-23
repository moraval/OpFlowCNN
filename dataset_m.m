function dataset_m()
rng(0)

datasize = 128;
% datasize = 1024;
% datasize = 10;
size1 = 64;
size2 = 64;

size1_sm = 16;
size2_sm = 16;

scale = 4;

data_big = zeros(size1,size2,6,datasize);
data_small = zeros(size1_sm, size2_sm, 6, datasize);
gt = zeros(size1_sm, size2_sm, 2, datasize);

for i=1:datasize
	data_big(:,:,1,i) = rand(size1,size2);
	
	for j=2:6
		data_big(:,:,j,i) = data_big(:,:,1,i);
	end

	data_big(:,:,:,i) = imgaussfilt(data_big(:,:,:,i),3);

	% normalizovat ty data 
	help = data_big(:,:,:,i);
	minb = min(help(:));
	maxb = max(help(:));
	data_big(:,:,:,i) = (help - minb) * 1/(maxb - minb);

	% create textured rectangle
	rec_size_x = 16;
	rec_size_y = 16;
	% rec_size_x = datasample([8 4],1);
	% rec_size_y = datasample([8 4],1);

	% numi = i + 0;
	% name = strcat('../dataset/textures/', num2str(numi), '.png');
	% texture = imread(name, 'png');
	% s1 = size(texture,1);
	% s2 = size(texture,2);
	% h1 = round(s1/2-rec_size_x/2);
	% h2 = round(s2/2-rec_size_y/2);
	% bg = zeros(rec_size_x, rec_size_y, 3);

	% if (size(texture,3) == 1)
	% 	help = texture(h1:h1 + rec_size_x-1, h2:h2 + rec_size_y-1);
	% 	bg = cat(3, help, help, help);
	% else
	% 	bg = texture(h1:h1 + rec_size_x-1, h2:h2 + rec_size_y-1, 1:3);
	% end
	% bg = double(bg) / 256;
	% imwrite(bg, strcat('background-', num2str(i), '.png'), 'png');
	
	if (rand < 0.5)
		bg_small = 0.75 + 0.25 * imgaussfilt(rand(rec_size_x, rec_size_y),1);
	else
		bg_small = 0 + 0.20 * imgaussfilt(rand(rec_size_x, rec_size_y),1);
	end
	bg = cat(3, bg_small, bg_small, bg_small);

	% randx = floor(randi([round(size1/2-rec_size_x*1.3) round(size1/2+rec_size_x/2.5)],1));
	% randy = floor(randi([round(size1/2-rec_size_y*1.3) round(size1/2+rec_size_y/2.5)],1));

	randx = floor(randi([round(size1/2-rec_size_x) round(size1/2)],1));
	randy = floor(randi([round(size1/2-rec_size_y) round(size1/2)],1));

	data_big(randx:randx+rec_size_x-1, randy:randy+rec_size_y-1,1:3,i) = bg;

	randx_small = round(randx/scale);
	randy_small = round(randy/scale);
	rec_size_x_sm = rec_size_x/scale;
	rec_size_y_sm = rec_size_y/scale;
	
	% shifts = nil
	shifts =  [-1*scale -1*scale; -1*scale 1*scale; 1*scale -1*scale; 1*scale 1*scale];
	% shifts =  [-1*scale 0; 0 1*scale; 1*scale 0; 0 -1*scale];
	shift = shifts(randi(4),:);
	% shift = shifts(randi(8),:);
	shiftx = shift(1);
	shifty = shift(2);
	randx = randx + shiftx;
	randy = randy + shifty;

	% shiftx = datasample([-1*scale 1*scale],1);
	% randx = randx + shiftx;
	% shifty = datasample([-1*scale 1*scale],1);
	% randy = randy + shifty;

	data_big(randx:randx+rec_size_x-1, randy:randy+rec_size_y-1,4:6,i) = bg;

	% size(gt(randx_small:randx_small+rec_size_x_sm-1, randy_small:randy_small+rec_size_y_sm-1,1,i))
	gt(randx_small:randx_small+rec_size_x_sm-1, randy_small:randy_small+rec_size_y_sm-1,1,i) = shiftx/scale*ones(rec_size_x_sm, rec_size_y_sm);
	gt(randx_small:randx_small+rec_size_x_sm-1, randy_small:randy_small+rec_size_y_sm-1,2,i) = shifty/scale*ones(rec_size_x_sm, rec_size_y_sm);

	data_small(:,:,:,i) = imresize(data_big(:,:,:,i),1/scale,'bilinear');

	% imwrite((gt(:,:,1,i)+1)/2, strcat('data/synt_mat/gt_1_', num2str(i),'.png'), 'png');
	% imwrite((gt(:,:,2,i)+1)/2, strcat('data/synt_mat/gt_2_', num2str(i),'.png'), 'png');

	% imwrite(data_big(:,:,1:3,i), strcat('data/synt_mat/img-test/', num2str(i), '_orig.png'), 'png');
	% imwrite(data_big(:,:,4:6,i), strcat('data/synt_mat/img-test/', num2str(i), '_target.png'), 'png');

	% imwrite(data_small(:,:,1:3,i), strcat('data/synt_mat/img-test/', num2str(i), '_orig_sm.png'), 'png');
	% imwrite(data_small(:,:,4:6,i), strcat('data/synt_mat/img-test/', num2str(i), '_target_sm.png'), 'png');
end

data_small(:,:,1,1);
data_big = permute(data_big, [4 3 1 2]);
data_small = permute(data_small, [4 3 1 2]);
gt = permute(gt, [4 3 1 2]);

% save('data/synt_mat/train-128-dataset-16-pix.mat','data_big','data_small','gt');
% save('data/synt_mat/validate-16-dataset-16-pix.mat','data_big','data_small','gt');
% save('data/synt_mat/test-big-data-16-pix_only-move_diff-bg.mat','data_big','data_small','gt');

% save('data/synt_mat/train-1024-dataset-restricted-4-16-pix.mat','data_big','data_small','gt');
save('data/synt_mat/validate-128-dataset-all-8-16-pix.mat','data_big','data_small','gt');


display 'done'
end
