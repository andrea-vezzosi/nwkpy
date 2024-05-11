
#############################################################################################
#############################################################################################
test_mesh_script="""
mesh Th0 = square(10 ,10);

mesh Th1 = square(4, 5);

real x0 = 1.2;
real x1 = 1.8;
real y0 = 0;
real y1 = 1;
int n = 5;
real m = 20;
mesh Th2 = square(n, m, [x0+(x1-x0)*x, y0+(y1-y0)*y]);
"""



Hex2reg_symm_ffem_script="""
// L = outer hexagon side length (in Angstrom)
real L = tw / sqrt(3.0) * 10.0;

// Lc = inner hexagon side length (in Angstrom)
real Lc = L - sw / sqrt(3.0) * 2.0 * 10.0;

// only the external border is generated with label = 1
border C1(t=0, 1){x=0.5*L*t; y=0.0;label=1;}

// while the other borders have label = 0
border C2(t=0, 1){x=0.5*L; y=(L-Lc)*(0.5*t*sqrt(3.0));label=0;}
border C3(t=0, 1){x=0.5*L; y=sqrt(3.0)*0.5*(L-Lc*(1.0-t));label=0;}
border C4(t=0, 1){x=0.5*(L-Lc+Lc*(1.0-t)); y=sqrt(3.0)*0.5*(L-Lc+Lc*(1.0-t));label=0;}
border C5(t=0, 1){x=0.5*(L-Lc)*(1.0-t); y=sqrt(3.0)*0.5*(L-Lc)*(1.0-t);label=0;}
border C6(t=0, 1){x=0.5*(L-Lc)+0.5*Lc*t; y=(L-Lc)*sqrt(3.0)*0.5;label=0;}

// create one half-wedge of the hexagon
mesh Th01 = buildmesh(C1(nC1) + C2(nC2) + C3(nC3) + C4(nC4) + C5(nC5) + C6(nC6));

// The origin (0,0) of the coordinate system is bottom-left corner
//
//           / |
//          /  |
//         4   3
//        /    |
//       /--6--|
//      5      2
//     /       | 
//  (0,0)---1--- 
//

// core has region number 1, shell has region number 2
int[int] reg = [0,1, 1,2];
Th01 = change(Th01, region=reg);

// symmetry operations

// reflection with respect to the y=0 plane
mesh Th02 = movemesh(Th01,[L-x,y]);

// obtain one full wedge
mesh Th1 = Th01 + Th02;

// rotate wedge 60° clockwise
mesh Th2 = movemesh(Th1,[L+(x*cos(pi/3.0)-y*sin(pi/3.0)),x*sin(pi/3.0)+y*cos(pi/3.0)]);

// rotate wedge 60° anticlockwise
mesh Th3 = movemesh(Th1,[-(x*cos(pi/3.0)-y*sin(pi/3.0)),(x*sin(pi/3.0)+y*cos(pi/3.0))]);

// define labels correspondance
int[int] r1=[1,1];
int[int] r2=[1,2];
int[int] r3=[1,6];

// bottom edge label = 1
Th1 = change(Th1, label=r1);

// bottom-left edge label = 2
Th2 = change(Th2, label=r2);

// bottom-right edge label = 6
Th3 = change(Th3, label=r3);

// obtain the lower half of the hexagon
mesh Thhalf1 = Th1 + Th2 + Th3;

// reflection with respect the plane x=0 gives upper half of the hexagon
mesh Thhalf2 = movemesh(Thhalf1,[x,L*sqrt(3.0)-y]);

// define labels correspondance
int[int] r4=[2,3, 1,4, 6,5];

// upper-right edge label = 3
// upper edge label = 4
// upper-left edge label = 5
Thhalf2 = change(Thhalf2, label=r4);

// obtain the full hexagon
mesh Thwhole = Thhalf1+Thhalf2;

// displace the mesh to have the origin (0,0) placed on the hexagon center
mesh Thwholed = movemesh(Thwhole,[x-L*0.5,y-tw*0.5*10.0]);

"""

#############################################################################################
#############################################################################################

Hex3reg_symm_ffem_script="""

real L = tw/sqrt(3.0) * 10.0;
real Lc = L - sw / sqrt(3.0) * 2.0 * 10.0;
real Lb = (tw + 2.0*bw) / sqrt(3.0) * 10.0;

border C1(t=0, 1){x=0.5*Lb*t; y=0.0;label=1;}
border C2(t=0, 1){x=0.5*Lb; y=(Lb-L)*(0.5*t*sqrt(3.0));label=0;}
border C3(t=0, 1){x=0.5*Lb; y=sqrt(3.0)*0.5*(Lb-L+(L-Lc)*t);label=0;}
border C4(t=0, 1){x=0.5*Lb; y=sqrt(3.0)*0.5*(Lb-Lc*(1.0-t));label=0;}
border C5(t=0, 1){x=0.5*(Lb-Lc+Lc*(1.0-t)); y=sqrt(3.0)*0.5*(Lb-Lc+Lc*(1.0-t));label=0;}
border C6(t=0, 1){x=0.5*(Lb-Lc+(Lc-L)*t); y=sqrt(3.0)*0.5*(Lb-Lc+(Lc-L)*t);label=0;}
border C7(t=0, 1){x=0.5*(Lb-L)*(1.0-t); y=sqrt(3.0)*0.5*(Lb-L)*(1.0-t);label=0;}
border C8(t=0, 1){x=0.5*(Lb-Lc)+0.5*Lc*t; y=(Lb-Lc)*sqrt(3.0)*0.5;label=0;}
border C9(t=0, 1){x=0.5*(Lb-L)+0.5*L*t; y=(Lb-L)*sqrt(3.0)*0.5;label=0;}

mesh Th01 = buildmesh(C1(nC1*0.5) + C2(nC2) + C3(nC3) + C4(nC4) + C5(nC5) + C6(nC6) + C7(nC7) + C8(-nC8*0.5) + C9(-nC9*0.5));
int[int] reg1 = [0,3, 1,2, 4,1 ];
Th01 = change(Th01, region=reg1);

// symmetry operations
mesh Th02 = movemesh(Th01,[Lb-x,y]);

mesh Th1 = Th01 + Th02;
mesh Th2 = movemesh(Th1,[Lb+(x*cos(pi/3.0)-y*sin(pi/3.0)),x*sin(pi/3.0)+y*cos(pi/3.0)]);
mesh Th3 = movemesh(Th1,[-(x*cos(pi/3.0)-y*sin(pi/3.0)),(x*sin(pi/3.0)+y*cos(pi/3.0))]);

int[int] r1=[1,1];
int[int] r2=[1,2];
int[int] r3=[1,6];
Th1 = change(Th1, label=r1);
Th2 = change(Th2, label=r2);
Th3 = change(Th3, label=r3);
mesh Thhalf1 = Th1 + Th2 + Th3;
mesh Thhalf2 = movemesh(Thhalf1,[x,Lb*sqrt(3.0)-y]);
int[int] r4=[2,3, 1,4, 6,5];
Thhalf2 = change(Thhalf2, label=r4);
mesh Thwhole = Thhalf1+Thhalf2;

mesh Thwholed = movemesh(Thwhole,[x-Lb*0.5,y-Lb*sqrt(3.0)*0.5]);

"""
