#!/usr/bin/perl

#------------#
# Parameters #
#------------#
$Num = 600;         # 抽出する文字数
$Chars = '[036]';   # 抽出する数字（任意の個数の数字を指定可能）


open IN1, "zcat train-labels.txt.gz 2>/dev/null |";
open IN2, "zcat train-images.txt.gz 2>/dev/null |";
open OUT1, ">sample-labels.txt";
open OUT2, ">sample-images.txt";

while($label = <IN1>) {
    $data = <IN2>;
    chomp $label;
    chomp $data;
    next unless ($label =~ m/$Chars/);
    $l = "";
    foreach $c (split(' ', $data)) {
        $l .= ($c>127?"1":"0").",";
    }
    chop $l;
    print OUT1 $label . "\n";
    print OUT2 $l . "\n";
    last if --$Num == 0;
}

close IN1;
close IN2;
close OUT1;
close OUT2;

open IN, "<sample-images.txt";
open OUT, ">samples.txt";
$c = 0;
while (<IN>) {
    $x = 0;
    for $s (split(/,/, $_)) {
        print OUT ($s==1?"#":" ");
        print OUT "\n" if ++$x % 28 == 0;
    } 
    last if ++$c == 10;
}
close IN;
close OUT;
