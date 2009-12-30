plot1 = fscanfMat('plot_tf.txt');
plot2 = fscanfMat('plot_der.txt');

fs = 4;
scf(1);
plot(plot1(:, 1), plot1(:, 2), '-'); 
plot(plot2(:, 1), plot2(:, 2), '--'); 