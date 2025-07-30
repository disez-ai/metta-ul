(: Lambda (-> Atom Atom (-> $a $t)))
(=
    ((Lambda $var $body) $arg)
    (let $var $arg $body)
)