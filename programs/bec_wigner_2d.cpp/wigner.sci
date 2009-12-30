plot1_1 = fscanfMat('plot1-1.dat');
plot1_2 = fscanfMat('plot1-2.dat');
plot2 = fscanfMat('plot2.dat');
plot3 = fscanfMat('plot3.dat');
plot4 = fscanfMat('plot4.dat');
plot5 = fscanfMat('plot5.dat');
plot6_1 = fscanfMat('plot6-1.dat');
plot6_2 = fscanfMat('plot6-2.dat');

fs = 4;

//scf(1);
//plot(plot1_1(:, 1), plot1_1(:, 2), '-');
//plot(plot1_2(:, 1), plot1_2(:, 2), '--'); //Plot mean number (a)
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N1', 'FontSize', fs);

//scf(2);
//plot(plot2(:, 1), plot2(:, 2), '--'); //Plot  variance
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('<Delta N1>', 'FontSize', fs);

//scf(3);
//plot(plot3(:, 1), plot3(:, 2), '--'); //Plot  mean number (b)
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N2', 'FontSize', fs);

//scf(4);
//plot(plot4(:, 1), plot4(:, 2), '--'); //Plot variance
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('<Delta N2>', 'FontSize', fs);

//scf(5);
//plot(plot5(:, 1), plot5(:, 2), '-'); //Plot total number vs time
//xlabel('t (ms)', 'FontSize', fs);
//ylabel('N', 'FontSize', fs);

scf(6);
//plot3d(plot6_1(1, 2:$), plot6_1(2:$, 1), plot6_1(2:$, 2:$));
grayplot(plot6_2(1, 2:$), plot6_2(2:$, 1), plot6_2(2:$, 2:$)); //Plot number density and mean density
xlabel('x (micron)', 'FontSize', fs);
ylabel('y (micron)', 'FontSize', fs);
zlabel('N', 'FontSize', fs);
