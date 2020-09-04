def res_block(inp, filters):
    x = Conv2D(filters, 3, padding='same', activation='relu')(inp)
    x = Conv2D(filters, 3, padding='same')(x)
    res = Add()([inp, x])
    return res
