"""Add to VerifData

Revision ID: 38c43395ea72
Revises: 58596a7a2414
Create Date: 2023-08-18 09:25:41.377342

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '38c43395ea72'
down_revision = '58596a7a2414'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('verif_features', sa.Column('selected_vars', sa.String(length=1000), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('verif_features', 'selected_vars')
    # ### end Alembic commands ###