mdp

module choices
  s : [0..3] init 0;

  [balanced] s=0 -> 1/5:(s'=1) + 3/10:(s'=2) + 1/2:(s'=3);
  [left]     s=0 -> 3/5:(s'=1) + 1/10:(s'=2) + 3/10:(s'=3);
  [] s=1 -> (s'=1);
  [] s=2 -> (s'=2);
  [] s=3 -> (s'=3);
endmodule

label "terminal_left" = s=1;
label "terminal_right" = s=2;
label "nontermination_or_other" = s=3;

