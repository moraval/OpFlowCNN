function dataset_m()

datasize = 16;
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

	% name = strcat('../dataset/textures/', num2str(i), '.jpg');
	% texture = imread(name);
	% s1 = size(texture,1);
	% s2 = size(texture,2);
	% h1 = round(s1/2-rec_size_x/1);
	% h2 = round(s2/2-rec_size_y/1);

	% bg = texture(h1:h1 + rec_size_x, h2:h2 + rec_size_y, :);
	% imwrite(bg, strcat('background-', num2str(i), '.png'));

	bg = datasample([0 1],1);
	
	randx = floor(randi([round(size1/2-rec_size_x*1.5) round(size1/2+rec_size_x/2)],1));
	randy = floor(randi([round(size1/2-rec_size_y*1.5) round(size1/2+rec_size_y/2)],1));

	for k = 1:rec_size_x
		for l = 1:rec_size_y
			 % data_big(randx+k, randy+l,1:3,i) = bg(k,l,:);
			 data_big(randx+k, randy+l,1:3,i) = 1;
		end
	end

	randx_small = round(randx/scale);
	randy_small = round(randy/scale);
	rec_size_x_sm = rec_size_x/scale;
	rec_size_y_sm = rec_size_y/scale;
	
	shiftx = datasample([-1*scale 1*scale],1);
	randx = randx + shiftx;
	shifty = datasample([-1*scale 1*scale],1);
	randy = randy + shifty;

	for k = 1:rec_size_x
		for l = 1:rec_size_y
			% data_big(randx+k, randy+l,4:6,i) = bg(k,l,:);
			data_big(randx+k, randy+l,4:6,i) = 1;
		end
	end

	for k = 1:rec_size_x_sm
		for l = 1:rec_size_y_sm
			gt(randx_small+k, randy_small+l,1:1,i) = shiftx/scale;
			gt(randx_small+k, randy_small+l,2:2,i) = shifty/scale;
		end
	end
	data_small(:,:,:,i) = imresize(data_big(:,:,:,i),1/scale,'bilinear');

	imwrite(data_big(:,:,1:3,i), strcat('data/synt_mat/img/', num2str(i), '_orig.png'));
	imwrite(data_big(:,:,4:6,i), strcat('data/synt_mat/img/', num2str(i), '_target.png'));

	imwrite(data_small(:,:,1:3,i), strcat('data/synt_mat/img/', num2str(i), '_orig_sm.png'));
	imwrite(data_small(:,:,4:6,i), strcat('data/synt_mat/img/', num2str(i), '_target_sm.png'));
end

data_small(:,:,1,1);
data_big = permute(data_big, [4 3 1 2]);
data_small = permute(data_small, [4 3 1 2]);
gt = permute(gt, [4 3 1 2]);

% save('data/synt_mat/train-128-dataset-16-pix.mat','data_big','data_small','gt');
save('data/synt_mat/validate-16-dataset-16-pix.mat','data_big','data_small','gt');
% save('data/synt_mat/test-big-data-16-pix_only-move_diff-bg.mat','data_big','data_small','gt');
% save('data/synt_mat/train-data-8-pix_all-dir_diff-bg_just-scaled.mat','data_big','data_small','gt');
% save('data/synt_mat/data-8x2-pix_all-dir_normed-bg.mat','data_big','data_small','gt');

display 'done'
end
