import datetime

from cd_mm_sits.constant.data import REF_DATE


def compute_delta_time(actual_date: datetime.date, base_date=REF_DATE) -> int:
    """Return the difference in days between
    a date and a reference date
    :param actual_date:
    :param base_date: YEAR-MONTH-DAY
    :returns:

    """
    year, month, day = base_date.split("-")
    base_date = datetime.date(int(year), int(month), int(day))
    delta = actual_date - base_date
    return delta.days


def from_int2date(int_date, ref_date: str | None = None):
    if ref_date is None:
        ref_date = REF_DATE
    year, month, day = ref_date.split("-")
    reference_date = datetime.datetime(int(year), int(month), int(day))
    return reference_date + +datetime.timedelta(int(int_date))
