function dataset_m()
%csvwrite(filename,M)

datasize = 32;
size1 = 64;
size2 = 64;

data_big = ones(size1,size2,6,datasize);
data_small = zeros(16, 16, 6, datasize);
gt = zeros(16, 16, 2, datasize);

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
	

	% help = imresize(data_big(:,:,1:3,i), 1/4);
	data_small(:,:,1:3,i) = imresize(data_big(:,:,1:3,i),1/4, 'bilinear');
	data_small(:,:,4:6,i) = imresize(data_big(:,:,4:6,i),1/4, 'bilinear');

	rec_size = 16;
	randx = round(randi([size1/2-rec_size/2 size1/2+rec_size/4],1));
	randy = round(randi([size1/2-rec_size/2 size1/2+rec_size/4],1));
	
	%data_big(randx:randx+rec_size,randy:randy+rec_size,1:3,i) = 1;
	help = 0.6 + 0.40 * imgaussfilt(rand(rec_size, rec_size),1);
	%help = 0.60 + 0.4 * (rand(rec_size, rec_size));
	help_small = imresize(help, 1/4, 'bilinear');

	for k = 1:rec_size
		for l = 1:rec_size
			data_big(randx+k, randy+l,1:3,i) = help(k,l);
			%data_big(randx+k, randy+l,1:3,i) = 1;
		end
	end
	
	shift = datasample([-4 4],1);
	shift = 4;

	randx_small = round(randx/4);
	randy_small = round(randy/4);
	rec_size_sm = rec_size/4;
	%data_small(randx_small:randx_small+rec_size_sm,randy_small:randy_small+rec_size_sm,1:3,i) = 1;
	for k = 1:rec_size_sm
		for l = 1:rec_size_sm
			data_small(randx_small+k, randy_small+l,1:3,i) = help_small(k,l);
			gt(randx_small+k, randy_small+l,1:2,i) = shift/4;
		end
	end

	randx = randx + shift;
	randy = randy + shift;
	%data_big(randx:randx+rec_size, randy:randy+rec_size,4:6,i) = 1;
	for k = 1:rec_size
		for l = 1:rec_size
			data_big(randx+k, randy+l,4:6,i) = help(k,l);
			%data_big(randx+k, randy+l,4:6,i) = 1;
		end
	end

	randx_small = round(randx/4);
	randy_small = round(randy/4);
	%data_small(randx_small:randx_small+rec_size_sm,randy_small:randy_small+rec_size_sm,4:6,i) = 1;
	for k = 1:rec_size_sm
		for l = 1:rec_size_sm
			data_small(randx_small+k, randy_small+l,4:6,i) = help_small(k,l);
		end
	end

	% display(strcat('data_synth/orig_', num2str(i), '.png'))
	imwrite(data_big(:,:,1:3,i), strcat('data/synt_mat/img/orig_', num2str(i), '.png'));
	imwrite(data_big(:,:,4:6,i), strcat('data/synt_mat/img/target_', num2str(i), '.png'));

	imwrite(data_small(:,:,1:3,i), strcat('data/synt_mat/img/orig_sm_', num2str(i), '.png'));
	imwrite(data_small(:,:,4:6,i), strcat('data/synt_mat/img/target_sm_', num2str(i), '.png'));
end

data_small(:,:,1,1);
data_big = permute(data_big, [4 3 1 2]);
data_small = permute(data_small, [4 3 1 2]);
gt = permute(gt, [4 3 1 2]);

% save('data/synt_mat/data-allnorm-biggerrect-twodir.mat','data_big','data_small','gt');

display 'done'
end


%data_big(randx:randx+rec_size, randy:randy+rec_size) = 1;

%csvwrite('big_data-2.csv', data_big);
%csvwrite('data_small-2.csv', data_small);