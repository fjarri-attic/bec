plot1 = fscanfMat('plot1.dat');
plot2 = fscanfMat('plot2.dat');
plot3 = fscanfMat('plot3.dat');
plot4 = fscanfMat('plot4.dat');
plot5 = fscanfMat('plot5.dat');
plot6 = fscanfMat('plot6.dat');

fs = 4;

scf(1);
plot(plot1(:, 1), plot1(:, 2), '--');
plot(plot1(:, 1), plot1(:, 3), '--'); //Plot mean number (a)
xlabel('t (ms)', 'FontSize', fs);
ylabel('N1', 'FontSize', fs);

scf(2);
plot(plot2(:, 1), plot2(:, 2), '--'); //Plot  variance
xlabel('t (ms)', 'FontSize', fs);
ylabel('<Delta N1>', 'FontSize', fs);

scf(3);
plot(plot3(:, 1), plot3(:, 2), '--'); //Plot  mean number (b)
xlabel('t (ms)', 'FontSize', fs);
ylabel('N2', 'FontSize', fs);

scf(4);
plot(plot4(:, 1), plot4(:, 2), '--'); //Plot variance
xlabel('t (ms)', 'FontSize', fs);
ylabel('<Delta N2>', 'FontSize', fs);

scf(5);
plot(plot5(:, 1), plot5(:, 2), '--'); //Plot total number vs time
xlabel('t (ms)', 'FontSize', fs);
ylabel('N', 'FontSize', fs);

scf(6);
plot(plot6(:, 1), plot6(:, 2), '--');
plot(plot6(:, 1), plot6(:, 3), '--'); //Plot number density and mean density
xlabel('z (micron)', 'FontSize', fs);
ylabel('N', 'FontSize', fs);
