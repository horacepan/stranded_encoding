pragma circom 2.0.4;

template Dot (n) {
    signal input a_enc[n];
    signal input b_enc[n];
    signal input d; // delta
    signal input exp[4];
    signal input prod;
    signal output result[4];

    // encoding the stuff happens outside of the circuit - evaluate the normal dot product, parse it in the circuit
    var d2 = d**2;
    var d6 = d**6;
    var d8 = d**8;

    signal tot;
    signal intermediate[n+1];
    intermediate[0] <== 0;
    for (var i = 0; i < n; i++) {
        intermediate[i+1] <== intermediate[i] + (a_enc[i] * b_enc[i]);
    }
    tot <== intermediate[n];
    tot === prod;

    // decode the result
    result[0] <-- tot % d;
    result[1] <-- (tot \ d2) % d;
    result[2] <-- (tot \ d6) % d;
    result[3] <-- (tot \ d8);

    result[0] === exp[0];
    result[1] === exp[1];
    result[2] === exp[2];
    result[3] === exp[3];
}

component main { public [ a_enc, b_enc, d, exp, prod] } = Dot(1000);
