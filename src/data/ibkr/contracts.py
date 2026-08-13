from __future__ import annotations

from typing import Any

from ibapi.contract import Contract


def stock_contract(
    symbol: str,
    currency: str = "USD",
    exchange: str = "SMART",
    primary_exchange: str | None = None,
) -> Contract:
    """Create an IBKR stock or ETF contract."""

    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = exchange
    contract.currency = currency

    if primary_exchange:
        contract.primaryExchange = primary_exchange

    return contract


def cash_contract(
    symbol: str,
    currency: str,
    exchange: str = "IDEALPRO",
) -> Contract:
    """
    Create an IBKR spot-forex contract.

    IBKR's CASH convention is symbol = base currency, currency = quote
    currency. For USD/JPY this is symbol="USD", currency="JPY", not
    symbol="USDJPY".
    """

    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "CASH"
    contract.exchange = exchange
    contract.currency = currency.upper()

    return contract


def crypto_contract(
    symbol: str,
    currency: str = "USD",
    exchange: str = "PAXOS",
) -> Contract:
    """Create an IBKR crypto contract."""

    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "CRYPTO"
    contract.exchange = exchange
    contract.currency = currency.upper()

    return contract


def commodity_contract(
    symbol: str,
    currency: str = "USD",
    exchange: str = "SMART",
) -> Contract:
    """Create an IBKR spot-commodity (e.g. spot metals) contract."""

    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "CMDTY"
    contract.exchange = exchange
    contract.currency = currency.upper()

    return contract


def index_contract(
    symbol: str,
    currency: str = "USD",
    exchange: str = "CBOE",
) -> Contract:
    """Create an IBKR index contract. Index contracts cannot be traded."""

    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "IND"
    contract.exchange = exchange
    contract.currency = currency.upper()

    return contract


CONTRACT_BUILDERS = {
    "STK": stock_contract,
    "CASH": cash_contract,
    "CRYPTO": crypto_contract,
    "CMDTY": commodity_contract,
    "IND": index_contract,
}


def build_contract(instrument_config: dict[str, Any]) -> Contract:
    """
    Dispatch to the correct contract builder based on instrument config.

    Expects `sec_type`, `ibkr_symbol`, `currency`, and `exchange` keys.
    """
    sec_type = instrument_config["sec_type"]

    try:
        builder = CONTRACT_BUILDERS[sec_type]
    except KeyError as error:
        available = sorted(CONTRACT_BUILDERS)

        raise ValueError(
            f"Unsupported sec_type '{sec_type}'. "
            f"Available: {available}"
        ) from error

    return builder(
        symbol=instrument_config["ibkr_symbol"],
        currency=instrument_config["currency"],
        exchange=instrument_config["exchange"],
    )
