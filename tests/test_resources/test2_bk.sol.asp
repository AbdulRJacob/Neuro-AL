in(A,B) :- A=img_1, B=circle_1.
circle(A) :- A=circle_1.
blue(A) :- A=circle_1.
image(A) :- A=img_1.
in(A,B) :- A=img_2, B=square_3.
in(A,B) :- A=img_2, B=circle_2.
blue(A) :- A=circle_2.
blue(A) :- A=square_3.
square(A) :- A=square_3.
circle(A) :- A=circle_2.
image(A) :- A=img_2.
c(A) :- alpha_44(A), image(A).
c_alpha_44(A) :- alpha_45(B,A), blue(B), in(A,B).
c_alpha_45(A,B) :- image(B), circle(A).

alpha_44(A) :- not c_alpha_44(A), image(A).
alpha_45(A,B) :- not c_alpha_45(A,B), blue(A), in(B,A).
