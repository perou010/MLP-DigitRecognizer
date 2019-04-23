function PlotZ2DScatter(Z,y)
hold on
colors = [
    [0 .447 .74];
    [0 0 1];
    [0 .5 0];
    [1 0 0];
    [0 .75 .75];
    [.75 0 .75];
    [.75 .75 0];
    [.25 .25 .25];
    [.3 .74 .933];
    [.5 .18 .57];
    ];
for n = 1:size(Z,1)
    color = colors(y(n)+1,:);
    scatter(Z(n,1), Z(n,2), 36, color);
    text(Z(n,1)+.005,Z(n,2)+.005,num2str(y(n)))
end
end