function PlotZ3DScatter(Z,y)
hold on
view(3);
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
for n = 1:(size(Z,1))
    color = colors(y(n)+1,:);
    scatter3(Z(n,1), Z(n,2),Z(n,3), 36, color);
    text(Z(n,1)+.005,Z(n,2)+.005,Z(n,3) +.005,num2str(y(n)))
end
end

