clear all
close all
datapath = '/Users/gq/tmp/ram/center/result/';
filename = 'center.gif';
%%
% pick_range_r = [59:433];
% pick_range_c = [144:514];

pick_range_r = [59:429];
pick_range_c = [140:516];
r_size = pick_range_r(end) - pick_range_r(1) + 1;
c_size = pick_range_c(end) - pick_range_c(1) + 1;

canvas = zeros(r_size, c_size * 10, 3);
h = figure(1); hold on
for step = 0:5
    for im_id = 1:10
        im = imread([datapath 'im_' num2str(im_id) '_step_' num2str(step) '.png']);
        canvas(1:r_size, 1 + c_size*(im_id-1):c_size + c_size*(im_id-1), :) = im(pick_range_r, pick_range_c, :);
    end
    imshow(canvas);
    set(gca,'position',[0 0 1 1],'units','normalized')
    axis equal; axis off;colormap gray
    frame = getframe(h);
    im = frame2im(frame);
    im = im(55: 260, :, :);
    
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File
    if step == 0
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
end