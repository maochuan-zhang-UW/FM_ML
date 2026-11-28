function h=plot_balloon(u1,u2,xx,yy,rr,scale,col);
%function h=plot_balloon(u1,u2,xx,yy,rr,scale,col);
% Adds a filled focal mechanism to current axes of radius RR and centered
% at XX,YY
% U1       Direction coordinates of normal to first plane (x=E, y=N, z=Up)
% U2       Direction coordinates of normal to second plane
% Specify U1 and U2 so that their sum is the direction of the T axes (Wilcock 2007)
% H are the handles
% SCALE Scales x axis by this factor (default1)
% COL - axis color
%
% Modified by Wilcock in 2007 to have revised plot_filled_mech.m method

if nargin<6
  scale=1;
end
if nargin<7
  col = 'k';
end

%Find line of points defining focal plane 1
if u1(1)==0; v=[1,0,0];
elseif u1(2)==0; v=[0,1,0];
else; v=[1 -u1(1)/u1(2) 0]; v=v/sqrt(sum(v.*v)); end
w=[u1(2)*v(3)-u1(3)*v(2) u1(3)*v(1)-u1(1)*v(3) u1(1)*v(2)-u1(2)*v(1)];
a=linspace(0.0000001,2*pi*0.9999999,360)';
vec=cos(a)*v+sin(a)*w;
vec=vec(vec(:,3)<=0,:);
az=atan2(vec(:,1),vec(:,2));
theta=acos(-vec(:,3));
p=sin(theta/2); x1=p.*sin(az); y1=p.*cos(az);

%Find line of points defining focal plane 2
if u2(1)==0; v=[1,0,0];
elseif u2(2)==0; v=[0,1,0];
else; v=[1 -u2(1)/u2(2) 0]; v=v/sqrt(sum(v.*v)); end
w=[u2(2)*v(3)-u2(3)*v(2) u2(3)*v(1)-u2(1)*v(3) u2(1)*v(2)-u2(2)*v(1)];
a=linspace(0.0000001,2*pi*0.9999999,360)';
vec=cos(a)*v+sin(a)*w;
vec=vec(vec(:,3)<=0,:);
az=atan2(vec(:,1),vec(:,2));
theta=acos(-vec(:,3));
p=sin(theta/2); x2=p.*sin(az); y2=p.*cos(az);

%Indicies i1,i2 of closest points in focal plane lines
smallest=1e99;
for i=1:length(x1)
  [small,j]=min((x2-x1(i)).^2+(y2-y1(i)).^2);
  if small<smallest
    i1=i;
    i2=j;
    smallest=small;
  end
end

%Make sure indicies i1,i2 are on lower side of intersection
if i1==1
elseif i1==length(x1)
  i1=i1-1;
elseif min((x2-x1(i1+1)).^2+(y2-y1(i1+1)).^2)>min((x2-x1(i1-1)).^2+(y2-y1(i1-1)).^2)
  i1=i1-1;
end
if i2==1
elseif i2==length(x2)
  i2=i2-1;
elseif min((x1-x2(i2+1)).^2+(y1-y2(i2+1)).^2)>min((x1-x2(i2-1)).^2+(y1-y2(i2-1)).^2)
  i2=i2-1;
end

%Find intersection xi,yi
A=[x1(i1)-x1(i1+1) x2(i2+1)-x2(i2); y1(i1)-y1(i1+1) y2(i2+1)-y2(i2)];
b=[x2(i2+1)-x1(i1+1); y2(i2+1)-y1(i1+1)]; 
w=inv(A)*b;
xi=w(1)*x1(i1)+(1-w(1))*x1(i1+1);
yi=w(1)*y1(i1)+(1-w(1))*y1(i1+1);

%Azimuths of start and end of each curve
a=zeros(1,4);
a(1) = atan2(x1(1),y1(1));
a(2) = atan2(x1(end),y1(end));
a(3) = atan2(x2(1),y2(1));
a(4) = atan2(x2(end),y2(end));

%Make 4 curves that enclose the quadrants
% Curve 1
xx1 = [x1(1:i1-1); xi; x2(i2:end)];
yy1 = [y1(1:i1-1); yi; y2(i2:end)];
a1 = a(1); 
a2 = a(4);
if a1>=a2 & (a1-a2)<=pi
  direc=1;
elseif a2>=a1 & (a2-a1)<=pi
  direc=2;
else
  if a1<0; a1=a1+2*pi; end;
  if a2<0; a2=a2+2*pi; end;
  if a1>=a2 & (a1-a2)<pi
    direc=1;
  elseif a2>=a1 & (a2-a1)<pi
    direc=2;
  else
    disp('plot_balloon - Should not get here 1')
    keyboard
  end
end
if direc==1
  xx1 = [xx1; sin(a2:pi/180:a1)'/sqrt(2)];
  yy1 = [yy1; cos(a2:pi/180:a1)'/sqrt(2)];
elseif direc==2
  xx1 = [xx1; sin(a2:-pi/180:a1)'/sqrt(2)];
  yy1 = [yy1; cos(a2:-pi/180:a1)'/sqrt(2)];
end
% Curve 2
xx2 = [x2(end:-1:i2); xi; x1(i1:end)];
yy2 = [y2(end:-1:i2); yi; y1(i1:end)];
a1 = a(4); 
a2 = a(2);
if a1>=a2 & (a1-a2)<=pi
  direc=1;
elseif a2>=a1 & (a2-a1)<=pi
  direc=2;
else
  if a1<0; a1=a1+2*pi; end;
  if a2<0; a2=a2+2*pi; end;
  if a1>=a2 & (a1-a2)<pi
    direc=1;
  elseif a2>=a1 & (a2-a1)<pi
    direc=2;
  else
    disp('plot_balloon - Should not get here 2')
    keyboard
  end
end
if direc==1
  xx2 = [xx2; sin(a2:pi/180:a1)'/sqrt(2)];
  yy2 = [yy2; cos(a2:pi/180:a1)'/sqrt(2)];
elseif direc==2
  xx2 = [xx2; sin(a2:-pi/180:a1)'/sqrt(2)];
  yy2 = [yy2; cos(a2:-pi/180:a1)'/sqrt(2)];
end
% Curve 3
xx3 = [x2(1:i2-1); xi; x1(i1:end)];
yy3 = [y2(1:i2-1); yi; y1(i1:end)];
a1 = a(3); 
a2 = a(2);
if a1>=a2 & (a1-a2)<=pi
  direc=1;
elseif a2>=a1 & (a2-a1)<=pi
  direc=2;
else
  if a1<0; a1=a1+2*pi; end;
  if a2<0; a2=a2+2*pi; end;
  if a1>=a2 & (a1-a2)<pi
    direc=1;
  elseif a2>=a1 & (a2-a1)<pi
    direc=2;
  else
    disp('plot_balloon - Should not get here 3')
    keyboard
  end
end
if direc==1
  xx3 = [xx3; sin(a2:pi/180:a1)'/sqrt(2)];
  yy3 = [yy3; cos(a2:pi/180:a1)'/sqrt(2)];
elseif direc==2
  xx3 = [xx3; sin(a2:-pi/180:a1)'/sqrt(2)];
  yy3 = [yy3; cos(a2:-pi/180:a1)'/sqrt(2)];
end
% Curve 4
xx4 = [x1(1:i1-1); xi; x2(i2-1:-1:1)];
yy4 = [y1(1:i1-1); yi; y2(i2-1:-1:1)];
a1 = a(1); 
a2 = a(3);
if a1>=a2 & (a1-a2)<=pi
  direc=1;
elseif a2>=a1 & (a2-a1)<=pi
  direc=2;
else
  if a1<0; a1=a1+2*pi; end;
  if a2<0; a2=a2+2*pi; end;
  if a1>=a2 & (a1-a2)<pi
    direc=1;
  elseif a2>=a1 & (a2-a1)<pi
    direc=2;
  else
    disp('plot_balloon - Should not get here 4')
    keyboard
  end
end
if direc==1
  xx4 = [xx4; sin(a2:pi/180:a1)'/sqrt(2)];
  yy4 = [yy4; cos(a2:pi/180:a1)'/sqrt(2)];
elseif direc==2
  xx4 = [xx4; sin(a2:-pi/180:a1)'/sqrt(2)];
  yy4 = [yy4; cos(a2:-pi/180:a1)'/sqrt(2)];
end
 


%Place 4 quadrants into x,y
n=[length(xx1) length(xx2) length(xx3) length(xx4)];
x(1:n(1),1)=xx1;
x(1:n(2),2)=xx2;
x(1:n(3),3)=xx3;
x(1:n(4),4)=xx4;
y(1:n(1),1)=yy1;
y(1:n(2),2)=yy2;
y(1:n(3),3)=yy3;
y(1:n(4),4)=yy4;
for i=1:4
  if n(i)<size(x,1)
    x(n(i)+1:end,i)=x(n(i),i);
    y(n(i)+1:end,i)=y(n(i),i);
  end
end

%T axis
ut=u1+u2;
if ut(3)>0; ut=-ut; end
ut = ut/sqrt(sum(ut.*ut));
xt = ut(1)/sqrt(2);
yt = ut(2)/sqrt(2);

%Find the curve that encloses the T axis
nsegment=zeros(1,4);
dx = x-xt;
dy = y-yt;
az = atan2(dx,dy);
for i=1:4
  for minaz = -pi:pi/12:pi-0.01;
    nsegment(i) = nsegment(i) + any(az(:,i)>=minaz & az(:,i)<=minaz+pi/12);
  end
end
[maxsegment,it] = max(nsegment);
if sum(nsegment==maxsegment)>1
  disp('plot_filled_mech.m - Tension axis in > 1 segment')
  %keyboard
end

if it==1 | it==3
  x=x(:,[1 3]);
  y=y(:,[1 3]);
else
  x=x(:,[2 4]);
  y=y(:,[2 4]);
end
x1 = x(:,1);
x2 = x(:,2);
y1 = y(:,1);
y2 = y(:,2);

%Plot it
if any(get(gca,'xdir')=='v'); x1=-x1; x2=-x2; end;
if any(get(gca,'ydir')=='v'); y1=-y1; y2=-y2; end;
a=linspace(0,2*pi,360);
h(1)=patch(sin(a)*rr*scale+xx,cos(a)*rr+yy,'w');
h(2)=line(sin(a)*rr*scale+xx,cos(a)*rr+yy,'color','k');
h(3)=patch(x1*sqrt(2)*rr*scale+xx,y1*sqrt(2)*rr+yy,col);
h(4)=patch(x2*sqrt(2)*rr*scale+xx,y2*sqrt(2)*rr+yy,col);
h=h(:)';
