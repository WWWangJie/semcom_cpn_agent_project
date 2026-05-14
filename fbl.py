import math


def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 10.0)


def q_function(x: float) -> float:
    """
    Gaussian Q-function using erfc.
    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def awgn_capacity_bits(snr_linear: float) -> float:
    """
    AWGN capacity in bits/channel use.
    """
    return math.log2(1.0 + snr_linear)


def channel_dispersion(snr_linear: float) -> float:
    """
    Channel dispersion for complex AWGN channel approximation.

    V = gamma * (gamma + 2) / (gamma + 1)^2 * (log2(e))^2
    """
    if snr_linear <= 0:
        return 1e-12
    log2e = math.log2(math.e)
    return (snr_linear * (snr_linear + 2.0) / ((snr_linear + 1.0) ** 2)) * (log2e ** 2)


def fbl_error_probability(
    payload_bits: int,
    blocklength: int,
    snr_db: float,
) -> float:
    """
    Finite blocklength block error probability.

    Parameters
    ----------
    payload_bits:
        Number of information bits.
    blocklength:
        Channel uses.
    snr_db:
        SNR in dB.

    Returns
    -------
    float
        Approximate block error probability in [0, 1].
    """
    if blocklength <= 0:
        raise ValueError("blocklength must be positive.")
    if payload_bits <= 0:
        return 0.0

    snr = db_to_linear(snr_db)
    rate = payload_bits / blocklength
    capacity = awgn_capacity_bits(snr)
    dispersion = max(channel_dispersion(snr), 1e-12)

    # Polyanskiy-type normal approximation.
    z = math.sqrt(blocklength / dispersion) * (capacity - rate) * math.log(2.0)
    eps = q_function(z)
    return float(min(max(eps, 0.0), 1.0))


def transmission_delay(payload_bits: int, spectral_efficiency: float, bandwidth_hz: float) -> float:
    """
    A lightweight transmission-time approximation.

    spectral_efficiency:
        Effective bit/s/Hz, usually lower than Shannon capacity.
    """
    rate_bps = max(spectral_efficiency * bandwidth_hz, 1.0)
    return payload_bits / rate_bps
