plot1 = fscanfMat('plot_tf.txt');
plot2 = fscanfMat('plot_gs.txt');
plot3 = fscanfMat('plot_plain.txt');

fs = 4;
scf(1);
plot(plot1(:, 1), plot1(:, 2), '-'); 
plot(plot2(:, 1), plot2(:, 2), '--');
plot(plot3(:, 1), plot3(:, 2), '-.');